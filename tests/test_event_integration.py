"""Panel events posted to the backend and broadcast to subscribed WebSocket clients.

``push_panel_event`` posts to the backend, which broadcasts to subscribed WebSocket clients.
The output shapes of the tools that push are part of the subject.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tcip_mcp.web_client import VALID_PANELS

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject
    from tcip_web.state import OpenProject


@pytest.fixture
def project_id(opened: OpenProject) -> str:
    """Return the open project's id."""
    return opened.id


# ── HTTP event bridge ────────────────────────────────────────────────────


class TestPostPanelEventRoute:
    """Tests of the FastAPI route that receives panel events from MCP tools."""

    def test_accepts_valid_panel(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        resp = opened_client.post(
            "/api/events/training",
            json={
                "project_id": project_id,
                "event_type": "metrics_update",
                "data": {"epoch": 5, "mAP50": 0.85},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["panel"] == "training"
        assert body["event_type"] == "metrics_update"

    def test_rejects_invalid_panel(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        resp = opened_client.post(
            "/api/events/bogus",
            json={
                "project_id": project_id,
                "event_type": "anything",
                "data": {},
            },
        )
        body = resp.json()
        assert "error" in body

    def test_all_valid_panels(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        # Iterates the shared set the tool and the route both validate against, so a panel added
        # there is covered here without a third copy of the list drifting out of step.
        for panel in sorted(VALID_PANELS):
            resp = opened_client.post(
                f"/api/events/{panel}",
                json={
                    "project_id": project_id,
                    "event_type": "test",
                    "data": {"ok": True},
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok", (
                f"panel {panel} should be valid"
            )

    def test_events_posted_while_connected_are_delivered_live_in_order(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """A connected subscriber receives events posted after it joined, in posting order."""
        with opened_client.websocket_connect(
            "ws://127.0.0.1/ws/panel/training"
        ) as ws:
            opened_client.post(
                "/api/events/training",
                json={
                    "project_id": project_id,
                    "event_type": "trial_update",
                    "data": {"trial": 1},
                },
            )
            opened_client.post(
                "/api/events/training",
                json={
                    "project_id": project_id,
                    "event_type": "trial_update",
                    "data": {"trial": 2},
                },
            )
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["data"] == {"trial": 1}
        assert second["data"] == {"trial": 2}

    def test_events_posted_before_connecting_are_replayed_on_connect(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """A browser connecting after events landed receives them on connect, in posting order."""
        panel = "results"
        opened_client.post(
            "/api/events/results",
            json={
                "project_id": project_id,
                "event_type": "count_ready",
                "data": {"count": 11},
            },
        )
        opened_client.post(
            "/api/events/results",
            json={
                "project_id": project_id,
                "event_type": "count_ready",
                "data": {"count": 22},
            },
        )

        with opened_client.websocket_connect(
            f"ws://127.0.0.1/ws/panel/{panel}"
        ) as ws:
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["event_type"] == "count_ready"
        assert first["data"] == {"count": 11}
        assert second["data"] == {"count": 22}

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

        opened_client.post(
            "/api/events/results",
            json={
                "project_id": project_id,
                "event_type": "count_ready",
                "data": {},
            },
        )
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
        and a ``count_ready`` event naming the open project is retained.
        """
        from tcip_web.state import store
        from tests._web_fixtures import named_project

        other = named_project(tmp_path.parent / "other", "Other")
        refused = opened_client.post(
            "/api/events/results",
            json={
                "project_id": other.id,
                "event_type": "count_ready",
                "data": {},
            },
        )
        assert refused.status_code == 409
        assert refused.json()["detail"]["open_project_id"] == project_id
        assert store.retained_events("results") == []

        admitted = opened_client.post(
            "/api/events/results",
            json={
                "project_id": project_id,
                "event_type": "count_ready",
                "data": {},
            },
        )
        assert admitted.status_code == 200, admitted.text
        assert len(store.retained_events("results")) == 1

    def test_annotate_focus_persists_advisory_state(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """An annotate_focus event writes its mode, subject and tab into the advisory state.

        An annotate_focus event carrying a mode and an active_subject writes both into the
        advisory state, alongside the tab it lands on.
        """
        resp = opened_client.post(
            "/api/events/app",
            json={
                "project_id": project_id,
                "event_type": "annotate_focus",
                "data": {
                    "subject": "bush",
                    "date": "2-11-26",
                    "mode": "polygon",
                    "active_subject": "bud",
                },
            },
        )
        assert resp.status_code == 200
        state = opened_client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "polygon"
        assert state["active_subject"] == "bud"

    def test_annotate_focus_with_an_unknown_mode_answers_400(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        before = opened_client.get("/api/state").json()["mode"]
        resp = opened_client.post(
            "/api/events/app",
            json={
                "project_id": project_id,
                "event_type": "annotate_focus",
                "data": {"mode": "lasso"},
            },
        )
        assert resp.status_code == 400
        assert "lasso" in resp.json()["detail"]
        assert opened_client.get("/api/state").json()["mode"] == before

    def test_the_focus_tool_answers_the_event_channels_refusal_whole(
        self,
        opened: OpenProject,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The focus tool answers the event channel's refusal whole.

        A focus the backend refuses carries the channel's reason and the project it has open,
        beside the requested image index.
        """
        from tcip_mcp import web_client
        from tcip_mcp.tools.gui_tools import focus_human_attention
        from tests._web_fixtures import bound_to

        refusal = {
            "error": "the backend does not have this project open",
            "open_project_id": "ffffffffffff",
            "delivered": False,
            "url": "u",
        }
        monkeypatch.setattr(
            web_client, "post_panel_event", lambda *a: dict(refusal)
        )

        res = focus_human_attention(
            bound_to(opened),
            str(data_dir),
            "bud",
            "2-11-26",
            mode="point",
            image_index=2,
        )

        assert {k: res[k] for k in refusal} == refusal
        assert res["image_index"] == 2

    def test_the_focus_tools_own_annotate_event_reaches_the_advisory_state(
        self,
        opened_client: TestClient,
        opened: OpenProject,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The event ``focus_human_attention`` posts sets the advisory state it names."""
        from tcip_mcp import web_client
        from tcip_mcp.tools.gui_tools import focus_human_attention
        from tests._web_fixtures import bound_to

        posted: dict = {}

        def _capture(
            target: BoundProject, panel: str, event_type: str, data: dict
        ) -> dict:
            posted.update(
                project_id=target.id,
                panel=panel,
                event_type=event_type,
                data=data,
            )
            return {"delivered": True, "status": "ok"}

        monkeypatch.setattr(web_client, "post_panel_event", _capture)
        res = focus_human_attention(
            bound_to(opened),
            str(data_dir),
            "bud",
            "2-11-26",
            mode="point",
            image_index=2,
        )
        assert "error" not in res, res
        assert posted["event_type"] == "annotate_focus"

        resp = opened_client.post(
            f"/api/events/{posted['panel']}",
            json={
                "project_id": posted["project_id"],
                "event_type": posted["event_type"],
                "data": posted["data"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        state = opened_client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "point"
        assert state["active_subject"] == "bud"


class TestPushPanelDataTool:
    """Tests of ``push_panel_event``, which posts over HTTP and answers without raising."""

    def test_no_subscribers_when_backend_down(
        self, bound: BoundProject, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no backend running, ``push_panel_event`` answers naming the panel."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
        result = push_panel_event(
            bound, "training", "metrics_update", {"epoch": 1}
        )
        # Either the connection was refused (no_subscribers) or a URL error;
        # both are acceptable. Tool must not raise.
        assert "status" in result or "error" in result
        # Panel name preserved in result
        assert result.get("panel") == "training"

    def test_invalid_panel_rejected(self, bound: BoundProject) -> None:
        """``push_panel_event`` answers an unknown panel name with an error."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        result = push_panel_event(bound, "bogus", "test", {})
        assert "error" in result


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
        """The port's record is written by ``web_client``'s one writer, which the backend calls."""
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
    """Tests that shared web records have their one key builder in ``web_client``."""

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

        from tcip_mcp.experiments import experiment_dir, find_run
        from tcip_mcp.tools import training_tools
        from tests._producer_fixtures import (
            fake_popen,
            seed_bud_images,
            small_detection_config,
        )

        images_dir = seed_bud_images(
            tmp_path / "images" / UNDATED_BUCKET,
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

        assert "error" not in res, res
        assert res["status"] == "launched"
        assert find_run(res["experiment_id"], project=tmp_path) == Path(
            res["output_dir"]
        )
        assert Path(res["output_dir"]) == experiment_dir(
            res["experiment_id"], project=tmp_path
        )
        assert res["pid"] == child.pid

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
        (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
        res = run_inference(
            tmp_path,
            ckpt,
            images_dir=str(tmp_path / "images" / UNDATED_BUCKET),
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
        images = tmp_path / "dataset" / "images" / "2026-01-01"
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
    res = post_panel_event(
        bound, "annotate", "annotate_focus", {"stem": "IMG_X"}
    )
    assert res == {
        "status": "suppressed_under_pytest",
        "delivered": False,
        "url": "",
    }


def test_post_panel_event_opt_in_bypasses_suppression(
    bound: BoundProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.web_client import post_panel_event, record_backend_port

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    record_backend_port(bound.workspace, 1)  # nothing listens on port 1
    res = post_panel_event(bound, "annotate", "annotate_focus", {})
    assert res["delivered"] is False
    assert (
        res["status"] != "suppressed_under_pytest"
    )  # it really attempted the send


def test_post_panel_event_answers_a_malformed_answer_as_an_error(
    bound: BoundProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listener answering with no HTTP status line is an error, never "no subscribers"."""
    import socketserver
    import threading

    from tcip_mcp.web_client import post_panel_event, record_backend_port

    class _NotHttp(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.recv(65536)
            self.request.sendall(b"NOT AN HTTP STATUS LINE\r\n\r\n")

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    with socketserver.TCPServer(("127.0.0.1", 0), _NotHttp) as server:
        record_backend_port(bound.workspace, server.server_address[1])
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        res = post_panel_event(bound, "annotate", "annotate_focus", {})
        thread.join(timeout=10)

    assert res["delivered"] is False
    assert res.get("status") != "no_subscribers"
    assert res["error"]


def test_post_panel_event_returns_the_backends_response_body(
    opened: OpenProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``post_panel_event`` answers a served backend's response body.

    Against a served backend it answers with the backend's response body; for another project
    than the open one, with the backend's structured refusal naming the open one; and after the
    project is closed, with the backend's own none-open refusal.
    """
    import asyncio

    from tcip_mcp.web_client import post_panel_event
    from tcip_web.app import app
    from tcip_web.state import store
    from tests._web_fixtures import bound_to, named_project
    from tools.smoke_terminal_e2e import served

    bound = bound_to(opened)
    other = bound_to(
        named_project(opened.root.parent / "hill_block", "Hill block")
    )
    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    with served(app, bound.workspace):
        res = post_panel_event(bound, "app", "status_note", {"text": "a note"})
        assert res["delivered"] is True
        assert res["response"] == {
            "status": "ok",
            "panel": "app",
            "event_type": "status_note",
        }

        elsewhere = post_panel_event(other, "app", "status_note", {})
        assert elsewhere["delivered"] is False
        assert elsewhere["open_project_id"] == opened.id
        assert elsewhere["error"]

        asyncio.run(store.close_project(opened.id))
        refused = post_panel_event(bound, "app", "status_note", {})
        assert refused["delivered"] is False
        assert "no project is open" in refused["error"]
