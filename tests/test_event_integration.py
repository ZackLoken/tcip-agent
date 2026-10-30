"""Panel events posted to the backend and broadcast to subscribed WebSocket clients.

``push_panel_event`` posts to the backend, which broadcasts to subscribed WebSocket clients.
The output shapes of the tools that push are part of the subject.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, get_args

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tcip_mcp.web_client import (
    AnnotateFocus,
    AnnotateMode,
    Banner,
    Panel,
    PanelEvent,
    PanelPost,
)

if TYPE_CHECKING:
    from starlette.testclient import httpx

    from tcip_mcp.workspace import BoundProject
    from tcip_web.state import OpenProject
    from tests._producer_fixtures import WrittenDataset


@pytest.fixture
def project_id(opened: OpenProject) -> str:
    """Return the open project's id."""
    return opened.id


def _focus(mode: AnnotateMode) -> AnnotateFocus:
    """Return a focus on the first image of a dated capture, in ``mode``."""
    return AnnotateFocus(
        dataset_root="/data",
        subject="bud",
        date="2-11-26",
        image_index=0,
        mode=mode,
        proposal=None,
    )


def _post(
    client: TestClient, panel: str, event: PanelEvent | dict, project_id: str
) -> httpx.Response:
    """Post ``event`` to ``panel`` for ``project_id``; a mapping is sent as it stands."""
    body = (
        {"event": event, "project_id": project_id}
        if isinstance(event, dict)
        else PanelPost(event=event, project_id=project_id).model_dump(
            mode="json"
        )
    )
    return client.post(f"/api/events/{panel}", json=body)


# ── HTTP event bridge ────────────────────────────────────────────────────


class TestPostPanelEventRoute:
    """Tests of the FastAPI route that receives panel events from MCP tools."""

    def test_accepts_valid_panel(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        resp = _post(
            opened_client, "training", Banner(text="epoch 5"), project_id
        )
        assert resp.status_code == 200
        assert resp.json() is None

    def test_a_path_naming_no_panel_is_refused_and_retained_nowhere(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """An event posted to a path naming no panel answers 422 and is retained nowhere."""
        from tcip_web.state import store

        resp = _post(opened_client, "bogus", Banner(text="x"), project_id)
        assert resp.status_code == 422
        assert all(
            store.retained_events(panel) == [] for panel in get_args(Panel)
        )

    def test_an_event_of_no_panel_event_type_is_refused_and_retained_nowhere(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """An event none of the four panel events is answers 422 and is retained nowhere."""
        from tcip_web.state import store

        resp = _post(
            opened_client,
            "training",
            {"event_type": "metrics_update", "epoch": 5},
            project_id,
        )
        assert resp.status_code == 422, resp.text
        assert store.retained_events("training") == []

    def test_a_socket_naming_no_panel_is_closed_with_1008(
        self, opened_client: TestClient
    ) -> None:
        with (
            pytest.raises(WebSocketDisconnect) as closed,
            opened_client.websocket_connect("ws://127.0.0.1/ws/panel/bogus"),
        ):
            pass
        assert closed.value.code == 1008

    def test_all_valid_panels(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        for panel in get_args(Panel):
            resp = _post(opened_client, panel, Banner(text="ok"), project_id)
            assert resp.status_code == 200, f"panel {panel} should be valid"

    def test_events_posted_while_connected_are_delivered_live_in_order(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """A connected subscriber receives events posted after it joined, in posting order."""
        with opened_client.websocket_connect(
            "ws://127.0.0.1/ws/panel/training"
        ) as ws:
            for text in ("trial 1", "trial 2"):
                _post(opened_client, "training", Banner(text=text), project_id)
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["event"] == Banner(text="trial 1").model_dump()
        assert second["event"] == Banner(text="trial 2").model_dump()

    def test_each_broadcast_drops_its_dead_socket_and_raises_any_other_send_fault(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """Each broadcast reaches its open socket and prunes a dead one; another fault raises.

        A state mutation and a panel event each reach their open socket and remove a socket whose
        transport is gone; a send failing for another reason raises and its socket stays.
        """
        import threading

        import anyio
        import anyio.from_thread
        from starlette.types import Message

        from tcip_web import app as web_app
        from tcip_web.state import store

        class _Peer:
            """A client whose sends raise ``fault`` once set; it hangs up when ``closed`` is set."""

            def __init__(self, closed: anyio.Event) -> None:
                self.closed = closed
                self.connected = False
                self.waiting = threading.Event()
                self.fault: Exception | None = None

            async def receive(self) -> Message:
                if not self.connected:
                    self.connected = True
                    return {"type": "websocket.connect"}
                self.waiting.set()
                await self.closed.wait()
                return {"type": "websocket.disconnect", "code": 1006}

            async def send(self, _message: Message) -> None:
                if self.fault is not None:
                    raise self.fault

        with anyio.from_thread.start_blocking_portal() as portal:
            closed = portal.call(anyio.Event)
            peers = [_Peer(closed), _Peer(closed), _Peer(closed)]
            watcher, subscriber, faulty = (
                WebSocket({"type": "websocket"}, peer.receive, peer.send)
                for peer in peers
            )
            handlers = [
                portal.start_task_soon(web_app.state_ws, watcher),
                portal.start_task_soon(
                    web_app.panel_ws, subscriber, "training"
                ),
                portal.start_task_soon(web_app.panel_ws, faulty, "meta"),
            ]
            try:
                assert all(peer.waiting.wait(10) for peer in peers)
                peers[0].fault = peers[1].fault = OSError("the peer is gone")
                peers[2].fault = RuntimeError(
                    "bound to a different event loop"
                )
                with (
                    opened_client.websocket_connect(
                        "ws://127.0.0.1/ws/state"
                    ) as state,
                    opened_client.websocket_connect(
                        "ws://127.0.0.1/ws/panel/training"
                    ) as panel,
                ):
                    state.receive_json()
                    for _ in store.retained_events("training"):
                        panel.receive_json()
                    opened_client.post(
                        "/api/state/tab", json={"active_tab": "training"}
                    )
                    _post(
                        opened_client,
                        "training",
                        Banner(text="trial 7"),
                        project_id,
                    )
                    pushed, event = state.receive_json(), panel.receive_json()
                    registered = (
                        watcher in web_app._state_watchers,
                        subscriber in web_app._panel_subscribers["training"],
                    )
                with pytest.raises(RuntimeError, match="different event loop"):
                    _post(
                        opened_client, "meta", Banner(text="note"), project_id
                    )
                assert faulty in web_app._panel_subscribers["meta"]
            finally:
                portal.call(closed.set)
                for handler in handlers:
                    handler.result(10)
                opened_client.post(
                    "/api/state/tab", json={"active_tab": "annotate"}
                )
        assert registered == (False, False)
        assert pushed["state"]["active_tab"] == "training"
        assert event["event"] == Banner(text="trial 7").model_dump()

    @pytest.mark.parametrize("route", ["state", "panel"])
    def test_a_socket_that_disconnects_mid_iteration_leaves_no_registration(
        self, opened_client: TestClient, route: str
    ) -> None:
        """A socket registered while it reads frames is unregistered once its peer disconnects."""
        import anyio
        from starlette.types import Message

        from tcip_web import app as web_app

        inbound: list[Message] = [
            {"type": "websocket.connect"},
            {"type": "websocket.receive", "text": "read and discarded"},
            {"type": "websocket.disconnect", "code": 1001},
        ]
        registered_while_reading: list[bool] = []

        async def receive() -> Message:
            if len(inbound) == 1:
                registered_while_reading.append(socket in registry)
            return inbound.pop(0)

        async def send(_message: Message) -> None:
            pass

        socket = WebSocket({"type": "websocket"}, receive, send)
        if route == "state":
            registry = web_app._state_watchers
            anyio.run(web_app.state_ws, socket)
        else:
            registry = web_app._panel_subscribers["training"]
            anyio.run(web_app.panel_ws, socket, "training")
        assert registered_while_reading == [True]
        assert socket not in registry

    def test_events_posted_before_connecting_are_replayed_on_connect(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """A browser connecting after events landed receives them on connect, in posting order."""
        panel = "results"
        for text in ("11 counted", "22 counted"):
            _post(opened_client, panel, Banner(text=text), project_id)

        with opened_client.websocket_connect(
            f"ws://127.0.0.1/ws/panel/{panel}"
        ) as ws:
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["event"] == Banner(text="11 counted").model_dump()
        assert second["event"] == Banner(text="22 counted").model_dump()

    def test_an_event_retained_for_one_project_is_not_replayed_once_another_is_open(
        self,
        opened_client: TestClient,
        project_id: str,
        tmp_path: Path,
    ) -> None:
        """Opening another project leaves none of the first project's events retained.

        One event retained for the first project leaves the store's retained results empty
        after another project opens.
        """
        from tcip_web.state import store
        from tests._web_fixtures import open_new_project

        _post(opened_client, "results", Banner(text="counted"), project_id)
        assert len(store.retained_events("results")) == 1

        open_new_project(tmp_path.parent / "other")
        assert store.retained_events("results") == []

    def test_an_event_naming_another_project_is_refused_and_retained_nowhere(
        self,
        opened_client: TestClient,
        project_id: str,
        tmp_path: Path,
    ) -> None:
        """An event naming another project is refused with 409 and retained nowhere.

        An event for another project answers 409 naming the open one and is retained nowhere,
        and the same event naming the open project is retained.
        """
        from tcip_web.state import store
        from tests._web_fixtures import named_project

        other = named_project(tmp_path.parent / "other", "Other")
        counted = Banner(text="counted")
        refused = _post(opened_client, "results", counted, other.id)
        assert refused.status_code == 409
        assert refused.json()["detail"]["open_project_id"] == project_id
        assert store.retained_events("results") == []

        admitted = _post(opened_client, "results", counted, project_id)
        assert admitted.status_code == 200, admitted.text
        assert len(store.retained_events("results")) == 1

    def test_annotate_focus_persists_advisory_state(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """An annotate_focus event writes its mode, its subject as the active one and the tab."""
        resp = _post(opened_client, "app", _focus(mode="polygon"), project_id)
        assert resp.status_code == 200
        state = opened_client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "polygon"
        assert state["active_subject"] == "bud"

    def test_annotate_focus_with_an_unknown_mode_answers_422(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        before = opened_client.get("/api/state").json()["mode"]
        resp = _post(
            opened_client,
            "app",
            {**_focus(mode="polygon").model_dump(), "mode": "lasso"},
            project_id,
        )
        assert resp.status_code == 422
        assert "lasso" in resp.text
        assert opened_client.get("/api/state").json()["mode"] == before

    def test_the_panel_socket_replays_no_refused_focus(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """A focus refused for its mode is replayed to no socket; the admitted one after it is.

        The socket's first frame is the admitted focus, carried as its model states it.
        """
        for mode in ("lasso", "point"):
            _post(
                opened_client,
                "app",
                {**_focus(mode="polygon").model_dump(), "mode": mode},
                project_id,
            )

        with opened_client.websocket_connect(
            "ws://127.0.0.1/ws/panel/app"
        ) as ws:
            replayed = ws.receive_json()

        assert replayed["event"] == _focus(mode="point").model_dump()

    def test_the_focus_tool_answers_the_event_channels_refusal_whole(
        self,
        opened: OpenProject,
        data_dir: WrittenDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The focus tool answers the event channel's refusal whole.

        A focus the backend refuses carries the channel's reason and the project it has open,
        beside the requested image index.
        """
        from tcip_mcp import web_client
        from tcip_mcp.tools.gui_tools import focus_human_attention
        from tcip_mcp.web_client import PanelEventAnswer, ProjectNotOpenRefusal
        from tests._web_fixtures import bound_to

        refusal = PanelEventAnswer(
            url="u",
            reason=ProjectNotOpenRefusal(
                message="the backend does not have this project open",
                open_project_id="ffffffffffff",
            ),
        )
        monkeypatch.setattr(web_client, "post_panel_event", lambda *a: refusal)

        res = focus_human_attention(
            bound_to(opened),
            str(data_dir.root),
            "bud",
            "2-11-26",
            mode="point",
            image_index=2,
        )

        answered = refusal.model_dump()
        assert {k: res[k] for k in answered} == answered
        assert res["image_index"] == 2

    def test_the_focus_tools_own_annotate_event_reaches_the_advisory_state(
        self,
        opened_client: TestClient,
        opened: OpenProject,
        data_dir: WrittenDataset,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The event ``focus_human_attention`` posts sets the advisory state it names."""
        from tcip_mcp import web_client
        from tcip_mcp.tools.gui_tools import focus_human_attention
        from tcip_mcp.web_client import PanelEventAnswer
        from tests._web_fixtures import bound_to

        posted: dict = {}

        def _capture(
            target: BoundProject, panel: Panel, event: PanelEvent
        ) -> PanelEventAnswer:
            posted.update(project_id=target.id, panel=panel, event=event)
            return PanelEventAnswer(url="u", reason=None)

        monkeypatch.setattr(web_client, "post_panel_event", _capture)
        res = focus_human_attention(
            bound_to(opened),
            str(data_dir.root),
            "bud",
            "2-11-26",
            mode="point",
            image_index=2,
        )
        assert "error" not in res, res
        assert isinstance(posted["event"], AnnotateFocus)

        resp = _post(
            opened_client,
            posted["panel"],
            posted["event"],
            posted["project_id"],
        )
        assert resp.status_code == 200, resp.text
        state = opened_client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "point"
        assert state["active_subject"] == "bud"


class TestPushPanelDataTool:
    """Tests of ``push_panel_event``, which posts over HTTP and answers without raising."""

    def test_no_subscribers_when_backend_down(
        self, bound: BoundProject, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no backend running, ``push_panel_event`` answers the reason it was not taken."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
        result = push_panel_event(bound, "training", Banner(text="epoch 1"))
        assert result.reason

    def test_the_server_refuses_a_name_outside_the_panels_and_admits_each_panel(
        self, bound: BoundProject
    ) -> None:
        """The tool's advertised schema names the panels; the server refuses any other name."""
        import asyncio

        from tests._server_calls import (
            called_through_the_server,
            client_server,
        )

        (listed,) = [
            t
            for t in asyncio.run(client_server(bound).list_tools())
            if t.name == "push_panel_event"
        ]
        assert listed.input_schema["properties"]["panel"]["enum"] == list(
            get_args(Panel)
        )
        arguments = {"event": Banner(text="test").model_dump()}
        refused = called_through_the_server(
            "push_panel_event", {**arguments, "panel": "bogus"}, bound=bound
        )
        assert isinstance(refused, str)
        assert "panel" in refused
        for panel in get_args(Panel):
            admitted = called_through_the_server(
                "push_panel_event", {**arguments, "panel": panel}, bound=bound
            )
            assert isinstance(admitted, dict)
            assert admitted["reason"] is not None


class TestPortDiscovery:
    """Tests of the client's read of the port the backend recorded under the workspace."""

    def test_the_recorded_port_is_read_whatever_the_environment_requests(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tcip_mcp.web_client import record_backend_port, resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        record_backend_port(tmp_path.parent, 34567)
        assert resolve_web_port(tmp_path.parent) == 34567

    def test_no_recorded_port_refuses_naming_what_records_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no recorded port, an environment port refuses, naming the recording command."""
        from tcip_mcp.web_client import NoBackendPortError, resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        with pytest.raises(NoBackendPortError, match="python -m tcip_web"):
            resolve_web_port(tmp_path.parent)

    def test_an_unreadable_recorded_port_raises_and_names_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recorded port that does not read as a port raises naming it, not read as absent."""
        import tcip_store as ts
        from tcip_mcp.web_client import backend_port_key, resolve_web_port

        ts.replace(backend_port_key(tmp_path.parent), "not a port")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        with pytest.raises(ValueError, match="valid integer"):
            resolve_web_port(tmp_path.parent)

    def test_the_port_handoff_is_one_declaration_that_both_packages_reach(
        self,
    ) -> None:
        """The backend's ``record_backend_port`` is ``web_client``'s own."""
        from tcip_mcp import web_client
        from tcip_web import __main__ as web_main

        assert web_main.record_backend_port is web_client.record_backend_port

    def test_the_backend_binds_and_the_tools_reach_one_loopback_host(
        self,
    ) -> None:
        from tcip_mcp import web_client
        from tcip_web import __main__ as web_main

        assert web_main.LOOPBACK_HOST is web_client.LOOPBACK_HOST
        assert web_client.LOOPBACK_HOST == "127.0.0.1"


class TestSharedWebStateDeclarations:
    """The backend reaches each shared web record through ``web_client``'s own functions."""

    def test_the_gui_snapshot_is_one_declaration_that_both_packages_reach(
        self,
    ) -> None:
        from tcip_mcp import web_client
        from tcip_web import state as web_state

        assert web_state.read_gui_snapshot is web_client.read_gui_snapshot
        assert web_state.write_gui_snapshot is web_client.write_gui_snapshot

    def test_the_canvas_documents_are_one_declaration_that_both_packages_reach(
        self,
    ) -> None:
        from tcip_mcp import web_client
        from tcip_web.routes import canvas

        assert canvas.canvas_meta_key is web_client.canvas_meta_key
        assert canvas.canvas_geometry_key is web_client.canvas_geometry_key

    def test_the_web_only_stores_are_addressed_through_web_client(
        self,
    ) -> None:
        """The sessions route's ``annotation_stats_key`` is ``web_client``'s own."""
        from tcip_mcp import web_client
        from tcip_web.routes import sessions

        assert sessions.annotation_stats_key is web_client.annotation_stats_key


# ── Tool output schemas ─────────────────────────────────────────────────


class TestTrainingToolOutputSchema:
    def test_launch_training_returns_the_experiment_id_its_artifacts_are_nested_under(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A launch answers its registered run's id and that run's own output directory."""
        pytest.importorskip("torchvision")
        monkeypatch.chdir(tmp_path)

        from tcip_mcp.experiments import (
            experiment_dir,
            named_directory,
            run_dirs,
        )
        from tcip_mcp.tools import training_tools
        from tests._producer_fixtures import (
            fake_popen,
            seed_bud_images,
            small_detection_config,
        )

        images_dir = seed_bud_images(
            image_dir(tmp_path, UNDATED_BUCKET),
            n=2,
            size=128,
            box=(10, 10, 40, 40),
        )

        child = fake_popen(monkeypatch, [])

        res = training_tools.launch_training(
            tmp_path,
            small_detection_config(images_dir, project=tmp_path),
            actor=None,
        )

        assert isinstance(res, training_tools.RunLaunched), res
        assert named_directory(res.experiment_id, run_dirs(tmp_path)) == Path(
            res.output_dir
        )
        assert Path(res.output_dir) == experiment_dir(
            res.experiment_id, project=tmp_path
        )
        assert res.pid == child.pid

    def test_monitor_training_answers_for_the_run_it_was_asked_about(
        self,
        tmp_path: Path,
    ) -> None:
        """``monitor_training`` answers for the run it was asked about.

        A status read names the run it describes and that run's own progress; an identifier
        naming no run is refused.
        """
        from tcip_mcp.tools import training_tools
        from tests._verified_checkpoint_fixtures import (
            detection_config,
            log_epoch,
            opened_run,
        )

        config = detection_config(tmp_path / "data", project=tmp_path)
        early = opened_run(tmp_path, config, experiment_id="event-run-early")
        log_epoch(early, 1, {"loss": 0.81})
        late = opened_run(tmp_path, config, experiment_id="event-run-late")
        log_epoch(late, 9, {"loss": 0.07})

        status = training_tools.monitor_training(tmp_path, late.name)["run"]
        assert status["experiment_id"] == late.name
        assert status["state"] == "running"
        assert status["current_epoch"] == 9
        assert status["output_dir"] == str(late)
        assert (
            training_tools.monitor_training(tmp_path, early.name)["run"][
                "current_epoch"
            ]
            == 1
        )

        assert "error" in training_tools.monitor_training(
            tmp_path, "run_that_was_never_created"
        )


class TestInferenceToolOutputSchema:
    def test_run_inference_reports_back_the_operating_point_it_was_handed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A dry run reports each operating-point dimension as its statement names it."""
        from tcip_mcp.pipelines.execution import Stated
        from tcip_mcp.tools.inference_tools import run_inference
        from tests._chain_fixtures import BLOB_BUILDER
        from tests._verified_checkpoint_fixtures import registered_checkpoint

        ckpt = registered_checkpoint(tmp_path, model_source=dict(BLOB_BUILDER))
        image_dir(tmp_path, UNDATED_BUCKET).mkdir(parents=True)
        res = run_inference(
            tmp_path,
            ckpt,
            images_dir=str(image_dir(tmp_path, UNDATED_BUCKET)),
            bucket="out/2026-01-01",
            dry_run=True,
            stated=Stated(
                tile=True,
                tile_size=512,
                overlap=0.35,
                conf=0.17,
                cross_tile_nms=0.55,
                postprocess="nmm",
                rationale="a sample reason the test states",
            ),
            actor=None,
        )

        assert "error" not in res, res
        assert res["dry_run"] is True
        execution = res["execution"]
        assert execution["conf"] == 0.17
        assert execution["tile_size"] == 512
        assert execution["overlap"] == 0.35
        assert execution["cross_tile_nms"] == 0.55
        assert execution["postprocess"] == "nmm"
        assert execution["sources"]["cross_tile_nms"] == "explicit"

    def test_a_published_bucket_holds_one_document_per_image_carrying_that_images_detections(
        self,
        tmp_path: Path,
    ) -> None:
        """A published bucket holds one document per image carrying that image's detections.

        A prediction bucket holds one document per image the pass saw, keyed by its stem and
        holding its own detections, an image with none included, and its record names each.
        """
        pytest.importorskip("torch")
        from tcip_annotation import json_io
        from tests._chain_fixtures import published

        def _boxes(n: int) -> list[list[float]]:
            return [
                [10.0 * i, 12.0 * i, 10.0 * i + 24.0, 12.0 * i + 18.0]
                for i in range(1, n + 1)
            ]

        counts = {"row3_plant07": 1, "row3_plant11": 0, "row9_plant02": 4}
        images = image_dir(tmp_path / "dataset", "2026-01-01")
        bucket = published(
            tmp_path,
            "baseline/2026-01-01",
            [
                {
                    "image": str(images / f"{stem}.jpg"),
                    "width": 800,
                    "height": 600,
                    "boxes": _boxes(n),
                    "scores": [0.9] * n,
                    "labels": [1] * n,
                    "cap": n + 1,
                }
                for stem, n in counts.items()
            ],
            scope={"subject": "bud"},
        )

        assert sorted(bucket.documents) == sorted(counts)
        written = {
            stem: len(
                json_io.read_predictions(
                    bucket.document_key(Path(source).stem)
                ).annotations
            )
            for stem, source in bucket.documents.items()
        }
        assert written == counts


class TestHpoToolOutputSchema:
    def test_run_hyperparameter_search_exists(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "run_hyperparameter_search")
        assert callable(training_tools.run_hyperparameter_search)


# ── Port fallback chain + pytest hermeticity ──────────


def test_post_panel_event_suppressed_under_pytest(
    bound: BoundProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``post_panel_event`` under pytest is suppressed and delivers nothing.

    Pytest itself sets PYTEST_CURRENT_TEST.
    """
    from tcip_mcp.web_client import post_panel_event

    monkeypatch.delenv("TCIP_ALLOW_PANEL_EVENTS", raising=False)
    res = post_panel_event(bound, "annotate", _focus(mode="box"))
    assert res.url is None
    assert (
        isinstance(res.reason, str) and "TCIP_ALLOW_PANEL_EVENTS" in res.reason
    )


def test_post_panel_event_opt_in_bypasses_suppression(
    bound: BoundProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.web_client import post_panel_event, record_backend_port

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    record_backend_port(bound.workspace, 1)  # nothing listens on port 1
    res = post_panel_event(bound, "annotate", _focus(mode="box"))
    assert isinstance(res.reason, str) and "no backend answered" in res.reason


def test_post_panel_event_answers_a_malformed_answer_as_an_error(
    bound: BoundProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listener answering with no HTTP status line is an error, never "no subscribers"."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from tcip_mcp.web_client import post_panel_event, record_backend_port

    class _NotHttp(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.wfile.write(b"NOT AN HTTP STATUS LINE\r\n\r\n")

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    with HTTPServer(("127.0.0.1", 0), _NotHttp) as server:
        record_backend_port(bound.workspace, server.server_address[1])
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        res = post_panel_event(bound, "annotate", _focus(mode="box"))
        thread.join(timeout=10)

    assert isinstance(res.reason, str)
    assert "no backend answered" not in res.reason


@pytest.mark.parametrize(
    "detail",
    [
        {"message": "a refusal naming no kind", "open_project_id": "p"},
        {
            "kind": "audit_entry_not_written",
            "message": "a refusal of another kind",
            "open_project_id": "p",
        },
        [
            {
                "type": "missing",
                "loc": ["body", "event"],
                "msg": "Field required",
            }
        ],
    ],
)
def test_post_panel_event_reads_a_detail_as_the_not_open_refusal_only_by_its_kind(
    bound: BoundProject, monkeypatch: pytest.MonkeyPatch, detail: object
) -> None:
    """A backend's detail of no kind, of another, or a validation list is answered as it was sent.

    Only a detail whose ``kind`` names the not-open refusal reads as one.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from tcip_mcp.web_client import post_panel_event, record_backend_port

    body = json.dumps({"detail": detail}).encode()

    class _Refusing(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    with HTTPServer(("127.0.0.1", 0), _Refusing) as server:
        record_backend_port(bound.workspace, server.server_address[1])
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        res = post_panel_event(bound, "annotate", _focus(mode="box"))
        thread.join(timeout=10)

    assert res.reason == detail


def test_post_panel_event_answers_the_delivery_and_the_backends_refusals(
    opened: OpenProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``post_panel_event`` answers a served backend's delivery and its refusals.

    Against a served backend it answers no reason; for another project than the open one, with
    the backend's structured refusal naming the open one; and after the project is closed, with
    the backend's own none-open refusal.
    """
    import asyncio

    from tcip_mcp.web_client import ProjectNotOpenRefusal, post_panel_event
    from tcip_web.app import app
    from tcip_web.state import store
    from tests._web_fixtures import bound_to, named_project
    from tools.smoke_terminal_e2e import served

    bound = bound_to(opened)
    other = bound_to(
        named_project(opened.root.parent / "hill_block", "Hill block")
    )
    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    note = Banner(text="a note")
    with served(app, bound.workspace):
        res = post_panel_event(bound, "app", note)
        assert res.reason is None

        elsewhere = post_panel_event(other, "app", note)
        assert isinstance(elsewhere.reason, ProjectNotOpenRefusal)
        assert elsewhere.reason.open_project_id == opened.id

        asyncio.run(store.close_project(opened.id))
        refused = post_panel_event(bound, "app", note)
        assert isinstance(refused.reason, str)
        assert "no project is open" in refused.reason
