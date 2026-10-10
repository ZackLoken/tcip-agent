"""``push_panel_event`` posting to the backend, which broadcasts to subscribed WebSocket clients,
and the output shapes of the tools that push."""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.web_client import VALID_PANELS


@pytest.fixture
def project_id(opened) -> str:
    """The open project's id, the one an event must name to be delivered."""
    return opened.id


# ── HTTP event bridge ────────────────────────────────────────────────────


class TestPostPanelEventRoute:
    """Verify the FastAPI stub route that receives events from MCP tools."""

    def test_accepts_valid_panel(self, opened_client: TestClient, project_id: str) -> None:
        resp = opened_client.post(
            "/api/events/training",
            json={
                "project_id": project_id, "event_type": "metrics_update",
                "data": {"epoch": 5, "mAP50": 0.85},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["panel"] == "training"
        assert body["event_type"] == "metrics_update"

    def test_rejects_invalid_panel(self, opened_client: TestClient, project_id: str) -> None:
        resp = opened_client.post(
            "/api/events/bogus",
            json={"project_id": project_id, "event_type": "anything", "data": {}},
        )
        body = resp.json()
        assert "error" in body

    def test_all_valid_panels(self, opened_client: TestClient, project_id: str) -> None:
        # Iterates the shared set the tool and the route both validate against, so a panel added
        # there is covered here without a third copy of the list drifting out of step.
        for panel in sorted(VALID_PANELS):
            resp = opened_client.post(
                f"/api/events/{panel}",
                json={"project_id": project_id, "event_type": "test", "data": {"ok": True}},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok", f"panel {panel} should be valid"

    def test_events_posted_while_connected_are_delivered_live_in_order(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """A subscriber connected to a panel receives events pushed after it joined, in the
        order they were posted."""
        with opened_client.websocket_connect("ws://127.0.0.1/ws/panel/training") as ws:
            opened_client.post(
                "/api/events/training",
                json={"project_id": project_id, "event_type": "trial_update", "data": {"trial": 1}},
            )
            opened_client.post(
                "/api/events/training",
                json={"project_id": project_id, "event_type": "trial_update", "data": {"trial": 2}},
            )
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["data"] == {"trial": 1}
        assert second["data"] == {"trial": 2}

    def test_events_posted_before_connecting_are_replayed_on_connect(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """A browser that connects after events landed receives them on connect, in the order
        they were posted."""
        panel = "results"
        opened_client.post(
            "/api/events/results",
            json={"project_id": project_id, "event_type": "count_ready", "data": {"count": 11}},
        )
        opened_client.post(
            "/api/events/results",
            json={"project_id": project_id, "event_type": "count_ready", "data": {"count": 22}},
        )

        with opened_client.websocket_connect(f"ws://127.0.0.1/ws/panel/{panel}") as ws:
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["event_type"] == "count_ready"
        assert first["data"] == {"count": 11}
        assert second["data"] == {"count": 22}

    def test_an_event_retained_for_one_project_is_not_replayed_once_another_is_open(
        self, opened_client: TestClient, project_id: str, tmp_path: Path,
    ) -> None:
        """Replay is of the open project's own events: after another project opens, a late
        subscriber is replayed nothing the first project's session retained."""
        from tcip_web.state import store
        from tests._web_fixtures import open_new_project

        opened_client.post("/api/events/results",
                    json={"project_id": project_id, "event_type": "count_ready", "data": {}})
        assert len(store.retained_events("results")) == 1

        open_new_project(tmp_path.parent / "other")
        assert store.retained_events("results") == []

    def test_an_event_naming_another_project_is_refused_and_retained_nowhere(
        self, opened_client: TestClient, project_id: str, tmp_path: Path,
    ) -> None:
        """The receiver admits the project an event names before retaining it: an event for
        another project answers 409 naming the open one and is retained nowhere, and the same
        event naming the open project is retained."""
        from tcip_web.state import store
        from tests._web_fixtures import named_project

        other = named_project(tmp_path.parent / "other", "Other")
        refused = opened_client.post("/api/events/results",
                    json={"project_id": other.id, "event_type": "count_ready", "data": {}})
        assert refused.status_code == 409
        assert refused.json()["detail"]["open_project_id"] == project_id
        assert store.retained_events("results") == []

        admitted = opened_client.post("/api/events/results",
                    json={"project_id": project_id, "event_type": "count_ready", "data": {}})
        assert admitted.status_code == 200, admitted.text
        assert len(store.retained_events("results")) == 1

    def test_annotate_focus_persists_advisory_state(
        self, opened_client: TestClient, project_id: str
    ) -> None:
        """An annotate_focus event carrying a mode and an active_subject writes both into the
        advisory state, alongside the tab it lands on."""
        resp = opened_client.post(
            "/api/events/app",
            json={
                "project_id": project_id,
                "event_type": "annotate_focus",
                "data": {
                    "subject": "bush", "date": "2-11-26", "mode": "polygon",
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
                "project_id": project_id, "event_type": "annotate_focus",
                "data": {"mode": "lasso"},
            },
        )
        assert resp.status_code == 400
        assert "lasso" in resp.json()["detail"]
        assert opened_client.get("/api/state").json()["mode"] == before

    def test_the_focus_tool_answers_the_event_channels_refusal_whole(
        self, opened, data_dir: Path, monkeypatch,
    ) -> None:
        """A focus the backend refuses carries the channel's reason and the project it has open,
        beside the frame it would have landed on."""
        from tcip_mcp import web_client
        from tcip_mcp.tools.gui_tools import focus_human_attention
        from tests._web_fixtures import bound_to

        refusal = {"error": "the backend does not have this project open",
                   "open_project_id": "ffffffffffff", "delivered": False, "url": "u"}
        monkeypatch.setattr(web_client, "post_panel_event", lambda *a: dict(refusal))

        res = focus_human_attention(bound_to(opened), str(data_dir), "bud", "2-11-26",
                                    mode="point", image_index=2)

        assert {k: res[k] for k in refusal} == refusal
        assert res["image_index"] == 2

    def test_the_focus_tools_own_annotate_event_reaches_the_advisory_state(
        self, opened_client: TestClient, opened, data_dir: Path, monkeypatch,
    ) -> None:
        """The event ``focus_human_attention`` posts, delivered to the backend, sets the
        advisory state it names."""
        from tcip_mcp import web_client
        from tcip_mcp.tools.gui_tools import focus_human_attention
        from tests._web_fixtures import bound_to

        posted: dict = {}

        def _capture(target, panel: str, event_type: str, data: dict) -> dict:
            posted.update(project_id=target.id, panel=panel, event_type=event_type, data=data)
            return {"delivered": True, "status": "ok"}

        monkeypatch.setattr(web_client, "post_panel_event", _capture)
        res = focus_human_attention(bound_to(opened), str(data_dir), "bud", "2-11-26",
                                    mode="point", image_index=2)
        assert "error" not in res, res
        assert posted["event_type"] == "annotate_focus"

        resp = opened_client.post(f"/api/events/{posted['panel']}",
                           json={"project_id": posted["project_id"],
                                 "event_type": posted["event_type"], "data": posted["data"]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        state = opened_client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "point"
        assert state["active_subject"] == "bud"


class TestPushPanelDataTool:
    """``push_panel_event`` posts over HTTP and answers without raising."""

    def test_no_subscribers_when_backend_down(self, bound, monkeypatch) -> None:
        """Backend not running → graceful 'no_subscribers' status."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
        result = push_panel_event(bound, "training", "metrics_update", {"epoch": 1})
        # Either the connection was refused (no_subscribers) or a URL error;
        # both are acceptable. Tool must not raise.
        assert "status" in result or "error" in result
        # Panel name preserved in result
        assert result.get("panel") == "training"

    def test_invalid_panel_rejected(self, bound) -> None:
        """Unknown panel names return an error before any HTTP call."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        result = push_panel_event(bound, "bogus", "test", {})
        assert "error" in result


class TestPortDiscovery:
    """The port the backend recorded under the workspace is the one place a client reads it."""

    def test_the_recorded_port_is_read_whatever_the_environment_requests(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import tcip_store as ts
        from tcip_mcp.web_client import backend_port_key, resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        ts.replace(backend_port_key(tmp_path.parent), "34567")
        assert resolve_web_port(tmp_path.parent) == 34567

    def test_no_recorded_port_refuses_naming_what_records_one(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """No record means no backend serves the workspace; a requested port in the environment
        is not where one listens."""
        from tcip_mcp.web_client import NoBackendPortError, resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        with pytest.raises(NoBackendPortError, match="python -m tcip_web"):
            resolve_web_port(tmp_path.parent)

    def test_an_unreadable_recorded_port_raises_and_names_it(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A handoff nothing can turn into a port is not a port, and is never read as absent."""
        import tcip_store as ts
        from tcip_mcp.web_client import backend_port_key, resolve_web_port

        ts.replace(backend_port_key(tmp_path.parent), "not a port")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        with pytest.raises(ValueError, match="does not read as a port"):
            resolve_web_port(tmp_path.parent)

    def test_the_port_handoff_is_one_declaration_that_both_packages_reach(self) -> None:
        """The port's key builder lives in ``web_client`` and the backend addresses it there."""
        from tcip_mcp import web_client
        from tcip_web import __main__ as web_main

        assert web_main.backend_port_key is web_client.backend_port_key

    def test_the_backend_binds_and_the_tools_reach_one_loopback_host(self) -> None:
        from tcip_mcp import web_client
        from tcip_web import __main__ as web_main

        assert web_main.LOOPBACK_HOST is web_client.LOOPBACK_HOST
        assert web_client.LOOPBACK_HOST == "127.0.0.1"


class TestSharedWebStateDeclarations:
    """Every record both packages touch has its one key builder in ``web_client``."""

    def test_the_gui_snapshot_is_one_declaration_that_both_packages_reach(self) -> None:
        from tcip_mcp import web_client
        from tcip_web import state as web_state

        assert web_state.read_gui_snapshot is web_client.read_gui_snapshot
        assert web_state.write_gui_snapshot is web_client.write_gui_snapshot

    def test_the_canvas_documents_are_one_declaration_that_both_packages_reach(self) -> None:
        from tcip_mcp import web_client
        from tcip_web.routes import canvas

        assert canvas.canvas_meta_key is web_client.canvas_meta_key
        assert canvas.canvas_geometry_key is web_client.canvas_geometry_key

    def test_the_web_only_stores_are_addressed_through_web_client(self) -> None:
        """The web routes address the web-only records through ``web_client``'s key builder."""
        from tcip_mcp import web_client
        from tcip_web.routes import sessions

        assert sessions.annotation_stats_key is web_client.annotation_stats_key


# ── Tool output schemas ─────────────────────────────────────────────────


class TestTrainingToolOutputSchema:
    def test_launch_training_returns_the_experiment_id_its_artifacts_are_nested_under(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The identifier a launch answers is the registered run, and its output directory is
        that run's own."""
        pytest.importorskip("torchvision")
        monkeypatch.chdir(tmp_path)

        from tcip_mcp.experiments import experiment_dir, find_run
        from tcip_mcp.pipelines.training import tensorboard_manager
        from tcip_mcp.tools import training_tools
        from tests._producer_fixtures import seed_bud_images, small_detection_config

        images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET, n=2, size=128,
                                     box=(10, 10, 40, 40))

        class _NoChild:
            """Stands in for the training subprocess; no child is spawned."""

            pid = 4242

            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(training_tools.subprocess, "Popen", _NoChild)
        monkeypatch.setattr(tensorboard_manager, "launch_tensorboard", lambda *a, **k: {})

        res = training_tools.launch_training(tmp_path, small_detection_config(images_dir),
                                             actor=None)

        assert "error" not in res, res
        assert res["status"] == "launched"
        assert find_run(res["experiment_id"], project=tmp_path) == Path(res["output_dir"])
        assert Path(res["output_dir"]) == experiment_dir(res["experiment_id"], project=tmp_path)
        assert res["pid"] == _NoChild.pid

    def test_monitor_training_answers_for_the_run_it_was_asked_about(
        self, tmp_path: Path,
    ) -> None:
        """A status read names the run it describes and that run's own progress; an identifier
        naming no run is refused."""
        from tcip_mcp.tools import training_tools
        from tests._verified_checkpoint_fixtures import detection_config, log_epoch, opened_run

        config = detection_config(tmp_path / "data")
        early = opened_run(tmp_path, config, experiment_id="event-run-early")
        log_epoch(early, 1, {"loss": 0.81})
        late = opened_run(tmp_path, config, experiment_id="event-run-late")
        log_epoch(late, 9, {"loss": 0.07})

        status = training_tools.monitor_training(tmp_path, late.name)["run"]
        assert status["experiment_id"] == late.name
        assert status["state"] == "running"
        assert status["current_epoch"] == 9
        assert status["output_dir"] == str(late)
        assert training_tools.monitor_training(tmp_path, early.name)["run"]["current_epoch"] == 1

        assert "error" in training_tools.monitor_training(tmp_path, "run_that_was_never_created")


class TestInferenceToolOutputSchema:
    def test_run_inference_reports_back_the_operating_point_it_was_handed(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A dry run reports each operating-point dimension as the caller named it."""
        from tcip_mcp.pipelines.execution import Stated
        from tcip_mcp.tools.inference_tools import run_inference
        from tests._chain_fixtures import BLOB_BUILDER
        from tests._verified_checkpoint_fixtures import registered_checkpoint

        ckpt = registered_checkpoint(tmp_path, model_source=dict(BLOB_BUILDER))
        (tmp_path / "images" / UNDATED_BUCKET).mkdir(parents=True)
        res = run_inference(tmp_path, ckpt, images_dir=str(tmp_path / "images" / UNDATED_BUCKET),
                            bucket="out/2026-01-01",
                            dry_run=True, stated=Stated(
                                tile=True, tile_size=512, overlap=0.35, conf=0.17,
                                cross_tile_nms=0.55, postprocess="nmm"))

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
        self, tmp_path: Path,
    ) -> None:
        """A prediction bucket holds one document per image the pass saw, keyed by its stem and
        holding its own detections, an image with none included, and its record names each."""
        pytest.importorskip("torch")
        from tcip_annotation import json_io

        from tests._chain_fixtures import published

        def _boxes(n: int) -> list[list[float]]:
            return [[10.0 * i, 12.0 * i, 10.0 * i + 24.0, 12.0 * i + 18.0] for i in range(1, n + 1)]

        counts = {"row3_plant07": 1, "row3_plant11": 0, "row9_plant02": 4}
        images = tmp_path / "dataset" / "images" / "2026-01-01"
        bucket = published(tmp_path, "baseline/2026-01-01", [
            {"image": str(images / f"{stem}.jpg"), "width": 800, "height": 600,
             "boxes": _boxes(n), "scores": [0.9] * n, "labels": [1] * n, "cap": n + 1}
            for stem, n in counts.items()], scope={"subject": "bud"})

        assert sorted(bucket.documents) == sorted(counts)
        written = {stem: len(json_io.read_predictions(
                       bucket.document_key(Path(source).stem)).annotations)
                   for stem, source in bucket.documents.items()}
        assert written == counts


class TestHpoToolOutputSchema:
    def test_run_hyperparameter_search_exists(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "run_hyperparameter_search")
        assert callable(training_tools.run_hyperparameter_search)


# ── Port fallback chain + pytest hermeticity ──────────


def test_post_panel_event_suppressed_under_pytest(bound, monkeypatch):
    """Test runs must never steer a live GUI (PYTEST_CURRENT_TEST is set by pytest itself)."""
    from tcip_mcp.web_client import post_panel_event

    monkeypatch.delenv("TCIP_ALLOW_PANEL_EVENTS", raising=False)
    res = post_panel_event(bound, "annotate", "annotate_focus", {"stem": "IMG_X"})
    assert res == {"status": "suppressed_under_pytest", "delivered": False, "url": ""}


def test_post_panel_event_opt_in_bypasses_suppression(bound, monkeypatch):
    from tcip_mcp.web_client import post_panel_event

    import tcip_store as ts
    from tcip_mcp.web_client import backend_port_key

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    ts.replace(backend_port_key(bound.workspace), "1")  # nothing listens on port 1
    res = post_panel_event(bound, "annotate", "annotate_focus", {})
    assert res["delivered"] is False
    assert res["status"] != "suppressed_under_pytest"   # it really attempted the send


def test_post_panel_event_returns_the_backends_response_body(opened, monkeypatch):
    """``post_panel_event`` against a served backend answers with the backend's response body,
    and with the backend's own none-open refusal once the project is closed."""
    import asyncio
    import socket
    import threading
    import time

    import uvicorn

    import tcip_store as ts
    from tcip_mcp.web_client import backend_port_key, post_panel_event
    from tcip_web.app import app
    from tcip_web.state import store
    from tests._web_fixtures import bound_to

    bound = bound_to(opened)
    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ts.replace(backend_port_key(bound.workspace), str(port))

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started, "the test backend never came up"

        res = post_panel_event(bound, "app", "status_note", {"text": "a note"})
        assert res["delivered"] is True
        assert res["response"] == {"status": "ok", "panel": "app", "event_type": "status_note"}

        asyncio.run(store.close_project(opened.id))
        refused = post_panel_event(bound, "app", "status_note", {})
        assert refused["delivered"] is False
        assert "no project is open" in refused["error"]
    finally:
        server.should_exit = True
        thread.join(timeout=10)
