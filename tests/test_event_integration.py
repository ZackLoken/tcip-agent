"""``push_panel_event`` posting to the backend, which broadcasts to subscribed WebSocket clients,
and the output shapes of the tools that push."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.web_client import VALID_PANELS
from tcip_web.app import app


@pytest.fixture
def client(opened_project):
    """A client of the web backend holding ``opened_project`` open."""
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def project_id(opened_project) -> str:
    """The open project's id, the one an event must name to be delivered."""
    from tcip_mcp.project_record import read_record

    return read_record(opened_project)["id"]


# ── HTTP event bridge ────────────────────────────────────────────────────


class TestPostPanelEventRoute:
    """Verify the FastAPI stub route that receives events from MCP tools."""

    def test_accepts_valid_panel(self, client: TestClient, project_id: str) -> None:
        resp = client.post(
            "/api/events/training",
            json={"project_id": project_id, "event_type": "metrics_update", "data": {"epoch": 5, "mAP50": 0.85}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["panel"] == "training"
        assert body["event_type"] == "metrics_update"

    def test_rejects_invalid_panel(self, client: TestClient, project_id: str) -> None:
        resp = client.post(
            "/api/events/bogus",
            json={"project_id": project_id, "event_type": "anything", "data": {}},
        )
        body = resp.json()
        assert "error" in body

    def test_all_valid_panels(self, client: TestClient, project_id: str) -> None:
        # Iterates the shared set the tool and the route both validate against, so a panel added
        # there is covered here without a third copy of the list drifting out of step.
        for panel in sorted(VALID_PANELS):
            resp = client.post(
                f"/api/events/{panel}",
                json={"project_id": project_id, "event_type": "test", "data": {"ok": True}},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok", f"panel {panel} should be valid"

    def test_events_posted_while_connected_are_delivered_live_in_order(
        self, client: TestClient, project_id: str
    ) -> None:
        """A subscriber connected to a panel receives events pushed after it joined, in the
        order they were posted."""
        with client.websocket_connect("ws://127.0.0.1/ws/panel/tuning") as ws:
            client.post(
                "/api/events/tuning",
                json={"project_id": project_id, "event_type": "trial_update", "data": {"trial": 1}},
            )
            client.post(
                "/api/events/tuning",
                json={"project_id": project_id, "event_type": "trial_update", "data": {"trial": 2}},
            )
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["data"] == {"trial": 1}
        assert second["data"] == {"trial": 2}

    def test_events_posted_before_connecting_are_replayed_on_connect(
        self, client: TestClient, project_id: str
    ) -> None:
        """A browser that connects after events landed receives them on connect, in the order
        they were posted."""
        panel = "results"
        client.post("/api/events/results",
                    json={"project_id": project_id, "event_type": "count_ready", "data": {"count": 11}})
        client.post("/api/events/results",
                    json={"project_id": project_id, "event_type": "count_ready", "data": {"count": 22}})

        with client.websocket_connect(f"ws://127.0.0.1/ws/panel/{panel}") as ws:
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["event_type"] == "count_ready"
        assert first["data"] == {"count": 11}
        assert second["data"] == {"count": 22}

    def test_an_event_retained_for_one_project_is_not_replayed_once_another_is_open(
        self, client: TestClient, project_id: str, tmp_path: Path,
    ) -> None:
        """Replay is of the open project's own events: after another project opens, a late
        subscriber is replayed nothing the first project's session retained."""
        from tcip_web.state import store
        from tests._web_fixtures import open_new_project

        client.post("/api/events/results",
                    json={"project_id": project_id, "event_type": "count_ready", "data": {}})
        assert len(store.retained_events("results")) == 1

        open_new_project(tmp_path.parent / "other")
        assert store.retained_events("results") == []

    def test_annotate_focus_persists_advisory_state(self, client: TestClient, project_id: str) -> None:
        """An annotate_focus event carrying a mode and an active_subject writes both into the
        advisory state, alongside the tab it lands on."""
        resp = client.post(
            "/api/events/app",
            json={
                "project_id": project_id,
                "event_type": "annotate_focus",
                "data": {"subject": "bush", "date": "2-11-26", "mode": "polygon", "active_subject": "bud"},
            },
        )
        assert resp.status_code == 200
        state = client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "polygon"
        assert state["active_subject"] == "bud"

    def test_annotate_focus_with_an_unknown_mode_answers_400(self, client: TestClient, project_id: str) -> None:
        before = client.get("/api/state").json()["mode"]
        resp = client.post(
            "/api/events/app",
            json={"project_id": project_id, "event_type": "annotate_focus", "data": {"mode": "lasso"}},
        )
        assert resp.status_code == 400
        assert "lasso" in resp.json()["detail"]
        assert client.get("/api/state").json()["mode"] == before

    def test_the_focus_tools_own_annotate_event_reaches_the_advisory_state(
        self, client: TestClient, opened_project: Path, project_id: str, data_dir: Path,
        monkeypatch,
    ) -> None:
        """The event ``focus_human_attention`` posts, delivered to the backend, sets the
        advisory state it names."""
        from tcip_mcp import web_client
        from tcip_mcp.tools.gui_tools import focus_human_attention

        posted: dict = {}

        def _capture(project: Path, workspace: Path, panel: str, event_type: str,
                     data: dict) -> dict:
            posted.update(panel=panel, event_type=event_type, data=data)
            return {"delivered": True, "status": "ok"}

        monkeypatch.setattr(web_client, "post_panel_event", _capture)
        res = focus_human_attention(opened_project, opened_project.parent,
                                    str(data_dir), "bud", "2-11-26", mode="point", image_index=2)
        assert "error" not in res, res
        assert posted["event_type"] == "annotate_focus"

        resp = client.post(f"/api/events/{posted['panel']}",
                           json={"project_id": project_id, "event_type": posted["event_type"],
                                 "data": posted["data"]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        state = client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "point"
        assert state["active_subject"] == "bud"


class TestPushPanelDataTool:
    """``push_panel_event`` posts over HTTP and answers without raising."""

    def test_no_subscribers_when_backend_down(self, project: Path, monkeypatch) -> None:
        """Backend not running → graceful 'no_subscribers' status."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
        monkeypatch.setenv("TCIP_WEB_PORT", "59999")  # very unlikely to be bound
        result = push_panel_event(project, project.parent, "training", "metrics_update",
                                  {"epoch": 1})
        # Either the connection was refused (no_subscribers) or a URL error;
        # both are acceptable. Tool must not raise.
        assert "status" in result or "error" in result
        # Panel name preserved in result
        assert result.get("panel") == "training"

    def test_invalid_panel_rejected(self, tmp_path: Path) -> None:
        """Unknown panel names return an error before any HTTP call."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        result = push_panel_event(tmp_path, tmp_path.parent, "bogus", "test", {})
        assert "error" in result


class TestPortDiscovery:
    """The record (the port actually bound) outranks the env var (a request), which outranks
    the default."""

    def test_record_wins_over_the_env_var(self, tmp_path: Path, monkeypatch) -> None:
        """The record names the port actually bound, so it outranks a request for a different one."""
        import tcip_store as ts
        from tcip_mcp.web_client import backend_port_key, resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        ts.replace(backend_port_key(tmp_path.parent), "34567")
        assert resolve_web_port(tmp_path.parent) == 34567

    def test_env_var_used_when_no_record_exists(self, tmp_path: Path, monkeypatch) -> None:
        """A failed publication or a bare ``uvicorn`` launch leaves no record: with none to trust,
        the request is the best information there is."""
        from tcip_mcp.web_client import resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        assert resolve_web_port(tmp_path.parent) == 12345

    def test_port_file_used_when_env_absent(self, tmp_path: Path, monkeypatch) -> None:
        import tcip_store as ts
        from tcip_mcp.web_client import backend_port_key, resolve_web_port

        ts.replace(backend_port_key(tmp_path.parent), "34567")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port(tmp_path.parent) == 34567

    def test_default_when_neither_available(self, tmp_path: Path, monkeypatch) -> None:
        from tcip_mcp.web_client import DEFAULT_PORT, resolve_web_port

        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port(tmp_path.parent) == DEFAULT_PORT

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
        """The port store is declared in ``web_client`` and the backend addresses it there."""
        import tcip_store as ts
        from tcip_mcp import web_client
        from tcip_web import __main__ as web_main

        assert web_main.backend_port_key is web_client.backend_port_key
        descriptor = ts.get_descriptor(web_client.BACKEND_PORT_STORE)
        assert descriptor.declared_in == web_client.__name__

    def test_the_backend_binds_and_the_tools_reach_one_loopback_host(self) -> None:
        from tcip_mcp import web_client
        from tcip_web import __main__ as web_main

        assert web_main.BACKEND_HOST is web_client.BACKEND_HOST
        assert web_client.BACKEND_HOST == "127.0.0.1"


class TestSharedWebStateDeclarations:
    """Every document both packages touch is declared once, in ``web_client``."""

    def test_the_gui_snapshot_is_one_declaration_that_both_packages_reach(self) -> None:
        import tcip_store as ts
        from tcip_mcp import web_client
        from tcip_web import state as web_state

        assert web_state.read_gui_snapshot is web_client.read_gui_snapshot
        assert web_state.write_gui_snapshot is web_client.write_gui_snapshot
        descriptor = ts.get_descriptor(web_client.GUI_SNAPSHOT_STORE)
        assert descriptor.declared_in == web_client.__name__

    def test_the_canvas_documents_are_one_declaration_that_both_packages_reach(self) -> None:
        import tcip_store as ts
        from tcip_mcp import web_client
        from tcip_web.routes import canvas

        assert canvas.canvas_meta_key is web_client.canvas_meta_key
        assert canvas.canvas_geometry_key is web_client.canvas_geometry_key
        for store in (web_client.CANVAS_META_STORE, web_client.CANVAS_GEOMETRY_STORE):
            assert ts.get_descriptor(store).declared_in == web_client.__name__

    def test_the_web_only_stores_are_declared_where_the_catalog_reaches_them(self) -> None:
        """The web-only stores are declared in ``web_client`` and the web routes address them
        there."""
        import tcip_store as ts
        from tcip_mcp import web_client
        from tcip_web.routes import sessions

        assert sessions.annotation_stats_key is web_client.annotation_stats_key
        assert ts.get_descriptor(web_client.ANNOTATION_STATS_STORE).declared_in == (
            web_client.__name__)


# ── Tool output schemas ─────────────────────────────────────────────────


class TestTrainingToolOutputSchema:
    def test_launch_training_returns_the_experiment_id_its_artifacts_are_nested_under(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The identifier a launch answers is the registered run, and its output directory is
        that run's own."""
        pytest.importorskip("torchvision")
        monkeypatch.chdir(tmp_path)

        from PIL import Image
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox
        from tcip_mcp.experiments import experiment_dir, find_run
        from tcip_mcp.pipelines.training import tensorboard_manager
        from tcip_mcp.tools import training_tools

        images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
        val_images, val_labels = tmp_path / "val_images", tmp_path / "val_labels"
        for d in (images_dir, labels_dir, val_images, val_labels):
            d.mkdir()
        for i in range(2):
            Image.new("RGB", (128, 128)).save(images_dir / f"t{i}.png")
            json_io.write_annotations(
                str(labels_dir / f"t{i}.json"),
                [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)
        Image.new("RGB", (128, 128)).save(val_images / "v0.png")
        json_io.write_annotations(
            str(val_labels / "v0.json"),
            [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 128, 128)

        class _NoChild:
            """Stands in for the training subprocess; no child is spawned."""

            pid = 4242

            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(training_tools.subprocess, "Popen", _NoChild)
        monkeypatch.setattr(tensorboard_manager, "launch_tensorboard", lambda *a, **k: {})

        cfg = {
            "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                             "builder_kwargs": {"min_size": 64, "max_size": 128},
                             "task": "detection"},
            "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                     "scope": {"subject": "bud"}},
            "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                         "mixed_precision": False, "device": "cpu",
        }
        res = training_tools.launch_training(tmp_path, cfg)

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

        status = training_tools.monitor_training(tmp_path, late.name)
        assert status["experiment_id"] == late.name
        assert status["status"] == "running"
        assert status["epoch"] == 9
        assert status["output_dir"] == str(late)
        assert training_tools.monitor_training(tmp_path, early.name)["epoch"] == 1

        assert "error" in training_tools.monitor_training(tmp_path, "run_that_was_never_created")


class TestInferenceToolOutputSchema:
    def test_run_inference_reports_back_the_operating_point_it_was_handed(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A dry run reports each operating-point dimension as the caller named it."""
        from tcip_mcp.pipelines.execution import Stated
        from tcip_mcp.tools.inference_tools import run_inference
        from tests._verified_checkpoint_fixtures import registered_checkpoint

        ckpt = registered_checkpoint(
            tmp_path, model_source={"builder": "tests.bespoke_models:build_bright_blob_detector",
                          "task": "detection"})
        res = run_inference(tmp_path, ckpt, images_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
                            dry_run=True, stated=Stated(
                                tile=True, tile_size=512, overlap=0.35, conf=0.17, max_dets=37,
                                cross_tile_nms=0.55, postprocess="nmm"))

        assert "error" not in res, res
        assert res["dry_run"] is True
        execution = res["execution"]
        assert execution["conf"] == 0.17
        assert execution["tile_size"] == 512
        assert execution["overlap"] == 0.35
        assert execution["max_dets"] == 37
        assert execution["cross_tile_nms"] == 0.55
        assert execution["postprocess"] == "nmm"
        assert execution["sources"]["cross_tile_nms"] == "explicit"

    def test_a_published_bucket_holds_one_file_per_image_carrying_that_images_detections(
        self, tmp_path: Path,
    ) -> None:
        """A prediction bucket holds one file per image the pass saw, named for its stem and
        holding its own detections, an image with none included, and its record names each."""
        pytest.importorskip("torch")
        from tests._chain_fixtures import published

        def _boxes(n: int) -> list[list[float]]:
            return [[10.0 * i, 12.0 * i, 10.0 * i + 24.0, 12.0 * i + 18.0] for i in range(1, n + 1)]

        counts = {"row3_plant07": 1, "row3_plant11": 0, "row9_plant02": 4}
        out = tmp_path / "dataset" / "predictions" / "baseline" / "2026-01-01"
        bucket = published(tmp_path, out, [
            {"image": f"{stem}.jpg", "width": 800, "height": 600, "boxes": _boxes(n),
             "scores": [0.9] * n, "labels": [1] * n}
            for stem, n in counts.items()], scope={"subject": "bud"})

        assert sorted(bucket.documents) == sorted(counts)
        written = {p.stem: len(json.loads(p.read_text())["annotations"])
                   for p in out.glob("*.json") if p.name != "bucket.json"}
        assert written == counts


class TestHpoToolOutputSchema:
    def test_run_hyperparameter_search_exists(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "run_hyperparameter_search")
        assert callable(training_tools.run_hyperparameter_search)


# ── Port fallback chain + pytest hermeticity ──────────


def test_post_panel_event_suppressed_under_pytest(tmp_path, monkeypatch):
    """Test runs must never steer a live GUI (PYTEST_CURRENT_TEST is set by pytest itself)."""
    from tcip_mcp.web_client import post_panel_event

    monkeypatch.delenv("TCIP_ALLOW_PANEL_EVENTS", raising=False)
    res = post_panel_event(tmp_path, tmp_path.parent, "annotate", "annotate_focus",
                           {"stem": "IMG_X"})
    assert res == {"status": "suppressed_under_pytest", "delivered": False, "url": ""}


def test_post_panel_event_opt_in_bypasses_suppression(project, monkeypatch):
    from tcip_mcp.web_client import post_panel_event

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    monkeypatch.setenv("TCIP_WEB_PORT", "1")        # nothing listens on port 1
    res = post_panel_event(project, project.parent, "annotate", "annotate_focus", {})
    assert res["delivered"] is False
    assert res["status"] != "suppressed_under_pytest"   # it really attempted the send


def test_post_panel_event_returns_the_backends_response_body(opened_project, monkeypatch):
    """``post_panel_event`` against a served backend answers with the backend's response body."""
    import socket
    import threading
    import time

    import uvicorn

    from tcip_mcp.web_client import post_panel_event
    from tcip_web.app import app

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("TCIP_WEB_PORT", str(port))

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

        res = post_panel_event(opened_project, opened_project.parent, "app", "status_note",
                               {"text": "a note"})
        assert res["delivered"] is True
        assert res["response"] == {"status": "ok", "panel": "app", "event_type": "status_note"}
    finally:
        server.should_exit = True
        thread.join(timeout=10)
