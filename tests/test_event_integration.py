"""Integration tests for the push_panel_event HTTP bridge and tool output schemas.

The legacy ``.tcip/events/`` file bridge has been retired. ``push_panel_event``
now POSTs to the tcip-web FastAPI backend; the backend broadcasts to any
subscribed WebSocket clients.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.web_client import VALID_PANELS
from tcip_web.app import app


@pytest.fixture
def client():
    return TestClient(app, base_url="http://127.0.0.1")


# ── HTTP event bridge ────────────────────────────────────────────────────


class TestPostPanelEventRoute:
    """Verify the FastAPI stub route that receives events from MCP tools."""

    def test_accepts_valid_panel(self, client: TestClient) -> None:
        resp = client.post(
            "/api/events/training",
            json={"event_type": "metrics_update", "data": {"epoch": 5, "mAP50": 0.85}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["panel"] == "training"
        assert body["event_type"] == "metrics_update"

    def test_rejects_invalid_panel(self, client: TestClient) -> None:
        resp = client.post(
            "/api/events/bogus",
            json={"event_type": "anything", "data": {}},
        )
        body = resp.json()
        assert "error" in body

    def test_all_valid_panels(self, client: TestClient) -> None:
        # Iterates the shared set the tool and the route both validate against, so a panel added
        # there is covered here without a third copy of the list drifting out of step.
        for panel in sorted(VALID_PANELS):
            resp = client.post(
                f"/api/events/{panel}",
                json={"event_type": "test", "data": {"ok": True}},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok", f"panel {panel} should be valid"

    def test_events_posted_while_connected_are_delivered_live_in_order(
        self, client: TestClient
    ) -> None:
        """A subscriber connected to a panel receives events pushed after it joined, in the
        order they were posted: the live half of what the deleted recent-events route served
        over HTTP to a reconnecting browser."""
        from tcip_web import app as web_app

        with client.websocket_connect("ws://127.0.0.1/ws/panel/tuning") as ws:
            for _ in range(len(web_app._recent_events.get("tuning", ()))):
                ws.receive_json()  # drain whatever earlier tests already posted to this panel
            client.post(
                "/api/events/tuning",
                json={"event_type": "trial_update", "data": {"trial": 1}},
            )
            client.post(
                "/api/events/tuning",
                json={"event_type": "trial_update", "data": {"trial": 2}},
            )
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["data"] == {"trial": 1}
        assert second["data"] == {"trial": 2}

    def test_events_posted_before_connecting_are_replayed_on_connect(
        self, client: TestClient
    ) -> None:
        """The ring buffer's whole reason to exist: a browser that connects after events
        already landed still sees them, in the order they were posted, replayed on the
        connection itself (the deleted GET recent-events route's job, now served by the
        on-connect loop at connect time rather than over a separate HTTP call)."""
        from tcip_web import app as web_app

        panel = "results"
        preexisting = len(web_app._recent_events.get(panel, ()))
        client.post("/api/events/results",
                    json={"event_type": "count_ready", "data": {"count": 11}})
        client.post("/api/events/results",
                    json={"event_type": "count_ready", "data": {"count": 22}})

        with client.websocket_connect(f"ws://127.0.0.1/ws/panel/{panel}") as ws:
            for _ in range(preexisting):
                ws.receive_json()  # drain whatever earlier tests already posted to this panel
            first = ws.receive_json()
            second = ws.receive_json()
        assert first["event_type"] == "count_ready"
        assert first["data"] == {"count": 11}
        assert second["data"] == {"count": 22}

    def test_review_focus_persists_advisory_state(self, client: TestClient) -> None:
        # The agent reads gui state back via view_gui_state: a focus event must
        # land there even though the browser applies it with local setters only.
        resp = client.post(
            "/api/events/app",
            json={
                "event_type": "review_focus",
                "data": {
                    "subject": "bud",
                    "date": "2-11-26",
                    "model_name": "m1",
                    "image_index": 3,
                    "detection_idx": 7,
                    "filter_type": "fp",
                    "iou_threshold": 0.4,
                    "conf_threshold": 0.3,
                },
            },
        )
        assert resp.status_code == 200
        state = client.get("/api/state").json()
        assert state["active_tab"] == "review"
        assert state["review"]["filter_type"] == "fp"
        assert state["review"]["detection_idx"] == 7
        assert state["review"]["iou_threshold"] == 0.4
        assert state["review"]["conf_threshold"] == 0.3

    def test_annotate_focus_persists_advisory_state(self, client: TestClient) -> None:
        """An annotate_focus event carrying a mode and an active_subject writes both into the
        advisory state, alongside the tab it lands on."""
        resp = client.post(
            "/api/events/app",
            json={
                "event_type": "annotate_focus",
                "data": {"subject": "bush", "date": "2-11-26", "mode": "polygon", "active_subject": "bud"},
            },
        )
        assert resp.status_code == 200
        state = client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "polygon"
        assert state["active_subject"] == "bud"

    def test_annotate_focus_with_an_unknown_mode_answers_400(self, client: TestClient) -> None:
        before = client.get("/api/state").json()["mode"]
        resp = client.post(
            "/api/events/app",
            json={"event_type": "annotate_focus", "data": {"mode": "lasso"}},
        )
        assert resp.status_code == 400
        assert "lasso" in resp.json()["detail"]
        assert client.get("/api/state").json()["mode"] == before

    def test_the_focus_tools_own_annotate_event_reaches_the_advisory_state(
        self, client: TestClient, data_dir: Path, monkeypatch,
    ) -> None:
        """The state an agent reads back is driven by the event the focus_human_attention tool
        really posts.

        The payload is taken from the producer rather than written here, so the two halves of the
        bridge are held to the same key names: a producer and a consumer that stop agreeing on
        what a field is called cannot both keep passing.
        """
        from tcip_mcp import web_client
        from tcip_mcp.tools.gui_tools import focus_human_attention

        from tests.test_canvas_liveview import _mint_binding

        posted: dict = {}

        def _capture(panel: str, event_type: str, data: dict) -> dict:
            posted.update(panel=panel, event_type=event_type, data=data)
            return {"delivered": True, "status": "ok"}

        monkeypatch.setattr(web_client, "post_panel_event", _capture)
        _mint_binding(data_dir)
        res = focus_human_attention("annotate", str(data_dir), str(data_dir), "bud", "2-11-26",
                                    mode="point", image_index=2)
        assert "error" not in res, res
        assert posted["event_type"] == "annotate_focus"

        resp = client.post(f"/api/events/{posted['panel']}",
                           json={"event_type": posted["event_type"], "data": posted["data"]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        state = client.get("/api/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "point"
        assert state["active_subject"] == "bud"


class TestActiveProjectChangedRoute:
    """The web backend treats the agent's adopt event as a signal to re-read the marker, not
    a name to trust: it repins to whatever the marker says, reports a disagreement with the
    event's own name, and never lets a repin failure block the broadcast."""

    def test_repins_this_process_from_the_marker(
        self, client: TestClient, tmp_path: Path, monkeypatch
    ) -> None:
        from tcip_mcp import workspace
        from tcip_mcp.project_paths import platform_state_root

        proj = workspace.project_path("chestnut_burr_valley")
        (proj / ".tcip").mkdir(parents=True)
        workspace.activate_project("chestnut_burr_valley")  # also repins this process, for now

        stale = tmp_path / "stale"
        stale.mkdir()
        monkeypatch.setenv("TCIP_STATE_ROOT", str(stale))
        assert platform_state_root() == stale

        resp = client.post(
            "/api/events/app",
            json={"event_type": "active_project_changed",
                  "data": {"name": "chestnut_burr_valley"}},
        )
        assert resp.status_code == 200
        assert resp.json()["platform_root"] == str(proj)
        assert platform_state_root() == proj

    def test_reports_a_disagreement_but_never_acts_on_the_events_own_name(
        self, client: TestClient
    ) -> None:
        from tcip_mcp import workspace

        proj = workspace.project_path("chestnut_burr_valley")
        (proj / ".tcip").mkdir(parents=True)
        workspace.activate_project("chestnut_burr_valley")

        resp = client.post(
            "/api/events/app",
            json={"event_type": "active_project_changed",
                  "data": {"name": "someone_elses_project"}},
        )
        body = resp.json()
        assert body["platform_root"] == str(proj)
        assert body["platform_root_disagreement"] == {
            "event_name": "someone_elses_project", "marker_name": "chestnut_burr_valley",
        }

    def test_reports_platform_root_problem_for_a_dangling_marker_and_still_broadcasts(
        self, client: TestClient
    ) -> None:
        import shutil

        from tcip_mcp import workspace
        from tcip_web import app as web_app

        proj = workspace.project_path("chestnut_burr_valley")
        (proj / ".tcip").mkdir(parents=True)
        workspace.activate_project("chestnut_burr_valley")
        shutil.rmtree(proj / ".tcip")

        with client.websocket_connect("ws://127.0.0.1/ws/panel/app") as ws:
            for _ in range(len(web_app._recent_events.get("app", ()))):
                ws.receive_json()  # drain whatever earlier tests already posted to this panel
            resp = client.post(
                "/api/events/app",
                json={"event_type": "active_project_changed",
                      "data": {"name": "chestnut_burr_valley"}},
            )
            body = resp.json()
            assert "chestnut_burr_valley" in body["platform_root_problem"]
            broadcast = ws.receive_json()
        assert broadcast["event_type"] == "active_project_changed"

    def test_an_event_naming_no_project_leaves_the_root_alone(
        self, client: TestClient
    ) -> None:
        from tcip_mcp.project_paths import platform_state_root

        before = platform_state_root()
        resp = client.post(
            "/api/events/app",
            json={"event_type": "active_project_changed", "data": {"name": "no_such_project"}},
        )
        body = resp.json()
        assert "platform_root" not in body
        assert "platform_root_problem" not in body
        assert platform_state_root() == before


class TestPushPanelDataTool:
    """Verify the MCP tool posts via HTTP and aliases legacy panel names."""

    def test_no_subscribers_when_backend_down(self, tmp_path: Path, monkeypatch) -> None:
        """Backend not running → graceful 'no_subscribers' status."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        from tests.test_canvas_liveview import _mint_binding

        # A matching binding, so the call reaches the HTTP push this exercises rather than being
        # refused by the binding rail before it.
        _mint_binding(tmp_path)
        # Point port discovery at an unused port in an isolated project root
        monkeypatch.setenv("TCIP_WEB_PORT", "59999")  # very unlikely to be bound
        result = push_panel_event(
            panel="training",
            event_type="metrics_update",
            data={"epoch": 1},
            project_root=str(tmp_path),
        )
        # Either the connection was refused (no_subscribers) or a URL error;
        # both are acceptable. Tool must not raise.
        assert "status" in result or "error" in result
        # Panel name preserved in result
        assert result.get("panel") == "training"

    def test_invalid_panel_rejected(self, tmp_path: Path) -> None:
        """Unknown panel names return an error before any HTTP call."""
        from tcip_mcp.tools.gui_tools import push_panel_event

        result = push_panel_event(panel="bogus", event_type="test", data={}, project_root=str(tmp_path))
        assert "error" in result


class TestPortDiscovery:
    """The record (the port actually bound) outranks the env var (a request), which outranks
    the default."""

    def test_record_wins_over_the_env_var(self, tmp_path: Path, monkeypatch) -> None:
        """The record names the port actually bound, so it outranks a request for a different one."""
        import tcip_store as ts
        from tcip_mcp.web_client import backend_port_key, resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        ts.replace(backend_port_key(), "34567")
        assert resolve_web_port() == 34567

    def test_env_var_used_when_no_record_exists(self, monkeypatch) -> None:
        """A failed publication or a bare ``uvicorn`` launch leaves no record: with none to trust,
        the request is the best information there is."""
        from tcip_mcp.web_client import resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        assert resolve_web_port() == 12345

    def test_port_file_used_when_env_absent(self, monkeypatch) -> None:
        import tcip_store as ts
        from tcip_mcp.web_client import backend_port_key, resolve_web_port

        ts.replace(backend_port_key(), "34567")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port() == 34567

    def test_the_port_the_backend_writes_is_the_port_a_tool_process_reads(
        self, monkeypatch,
    ) -> None:
        """The backend's own writer and the MCP-side resolver meet at one file.

        Both halves run here rather than the file being hand-written, so a tool in another process
        finds the port the backend actually bound instead of quietly falling back to the default.
        """
        from tcip_mcp.web_client import DEFAULT_PORT, resolve_web_port
        from tcip_web.__main__ import _write_port_file

        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        bound = 41871
        assert bound != DEFAULT_PORT
        _write_port_file(bound)
        assert resolve_web_port() == bound

    def test_port_record_found_regardless_of_this_process_own_platform_root(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The port handoff hangs off the workspace root, not the platform-state root a
        project adopts into, so a reader pinned to a different project than the one active
        when the backend started still finds the port it bound."""
        import tcip_store as ts
        from tcip_mcp import web_client

        ts.replace(web_client.backend_port_key(), "23456")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "some_other_project"))
        assert web_client.resolve_web_port() == 23456

    def test_default_when_neither_available(self, monkeypatch) -> None:
        from tcip_mcp.web_client import DEFAULT_PORT, resolve_web_port

        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port() == DEFAULT_PORT

    def test_an_unreadable_recorded_port_falls_through_to_the_default(
        self, monkeypatch
    ) -> None:
        """A handoff nothing can turn into a port is not a port.

        This lookup runs before the backend is known to be up, so it reports the default and
        lets the caller try rather than raising on a record it cannot use.
        """
        import tcip_store as ts
        from tcip_mcp.web_client import DEFAULT_PORT, backend_port_key, resolve_web_port

        ts.replace(backend_port_key(), "not a port")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port() == DEFAULT_PORT

    def test_the_port_handoff_is_one_declaration_that_both_packages_reach(self) -> None:
        """The reader owns the declaration and the backend imports it.

        The reader is in the MCP package and cannot import the web package, so a declaration on
        each side would be two stores wearing one name and whichever imported first would decide
        where the handoff lands.
        """
        import tcip_store as ts
        from tcip_mcp import web_client
        from tcip_web import __main__ as web_main

        assert web_main.backend_port_key is web_client.backend_port_key
        descriptor = ts.get_descriptor(web_client.BACKEND_PORT_STORE)
        assert descriptor.declared_in == web_client.__name__

    def test_host_env_override(self, monkeypatch) -> None:
        from tcip_mcp.web_client import resolve_web_host

        monkeypatch.setenv("TCIP_WEB_HOST", "10.0.0.1")
        assert resolve_web_host() == "10.0.0.1"


class TestSharedWebStateDeclarations:
    """Every document both packages touch is declared once, where the MCP side can import it.

    An MCP tool reads each of these and cannot import ``tcip_web``, so a declaration on each
    side would be two stores wearing one name and whichever imported first would decide where
    the document lands.
    """

    def test_the_gui_snapshot_is_one_declaration_that_both_packages_reach(self) -> None:
        import tcip_store as ts
        from tcip_mcp import web_client
        from tcip_web import state as web_state

        assert web_state.gui_snapshot_key is web_client.gui_snapshot_key
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


# ── Tool output schemas (unchanged from pre-HTTP migration) ─────────────


class TestTrainingToolOutputSchema:
    def test_launch_training_returns_the_run_id_its_artifacts_are_nested_under(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The identifier a launch hands back is the run the platform registered, and the
        directory that run writes into is that identifier's own.

        A caller holds one string afterwards; polling status with it and reading metrics under it
        have to reach the same run, so the two are checked against the registry rather than
        against each other.
        """
        pytest.importorskip("torchvision")
        monkeypatch.chdir(tmp_path)

        from PIL import Image
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox
        from tcip_mcp.pipelines.training import tensorboard_manager
        from tcip_mcp.pipelines.training.run_registry import get_run
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
            """Stands in for the training subprocess: this test is about what the launch reports,
            not about the training body, so no child is spawned."""

            pid = 4242

            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(training_tools.subprocess, "Popen", _NoChild)
        monkeypatch.setattr(tensorboard_manager, "launch_tensorboard", lambda *a, **k: {})

        cfg = {
            "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                             "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                             "task": "detection"},
            "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                     "subject": "bud", "val_images_dir": str(val_images),
                     "val_labels_dir": str(val_labels)},
            "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                         "mixed_precision": False, "device": "cpu"},
        }
        res = training_tools.launch_training(cfg, str(tmp_path / "runs"))

        assert "error" not in res, res
        assert res["status"] == "launched"
        registered = get_run(res["run_id"])
        assert registered is not None, f"no run registered under {res['run_id']!r}"
        assert Path(res["output_dir"]) == tmp_path / "runs" / res["run_id"]
        assert res["pid"] == _NoChild.pid

    def test_monitor_training_answers_for_the_run_it_was_asked_about(
        self, tmp_path: Path,
    ) -> None:
        """A status read carries the identifier of the run it describes, plus that run's own
        progress, so a caller tracking several runs at once can tell the answers apart. An
        identifier that names no run is refused rather than answered for some other run.
        """
        from tcip_mcp.pipelines.training.run_registry import create_run
        from tcip_mcp.tools import training_tools

        early = create_run({"seed": 11}, str(tmp_path / "early"))
        early.status, early.current_epoch, early.best_metric = "running", 1, 0.81
        late = create_run({"seed": 12}, str(tmp_path / "late"))
        late.status, late.current_epoch, late.best_metric = "completed", 9, 0.07

        status = training_tools.monitor_training(late.run_id)
        assert status["run_id"] == late.run_id
        assert status["run_id"] != early.run_id
        assert status["status"] == "completed"
        assert status["epoch"] == 9
        assert status["best_metric"] == 0.07
        assert status["output_dir"] == str(tmp_path / "late")

        assert "error" in training_tools.monitor_training("run_that_was_never_created")


class TestInferenceToolOutputSchema:
    def test_run_inference_reports_back_the_operating_point_it_was_handed(
        self, tmp_path: Path,
    ) -> None:
        """A dry run reports the operating point a real pass would measure at, each dimension as
        the caller named it.

        Every value here is distinct, so a dimension reported in another's place is visible rather
        than hidden behind two fields that happen to share a default.
        """
        from tcip_mcp.tools.inference_tools import run_inference

        ckpt = tmp_path / "detector.pt"
        ckpt.write_bytes(b"a dry run never loads the weights")
        res = run_inference(str(ckpt), images_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
                            dry_run=True, tile=True,
                            tile_size=512, overlap=0.35, conf_threshold=0.17, max_dets=37,
                            global_nms_iou=0.55, postprocess="nmm")

        assert "error" not in res, res
        assert res["dry_run"] is True
        assert res["checkpoint_path"] == str(ckpt)
        op = res["operating_point"]
        assert op["conf"] == 0.17
        assert op["tile_size"] == 512
        assert op["overlap"] == 0.35
        assert op["max_dets"] == 37
        assert op["cross_tile_nms"] == 0.55
        assert op["postprocess"] == "nmm"
        assert op["tiled"] is True
        assert op["tiled_source"] == "explicit"

    def test_run_inference_writes_one_file_per_image_carrying_that_images_detections(
        self, tmp_path: Path, monkeypatch, seed_bud_trait_spec,
    ) -> None:
        """A prediction bucket holds one file per image the pass saw, named for that image's own
        stem and holding that image's own detections.

        The three images here have three different detection counts, one of them zero, so a bucket
        that pairs a file to the wrong image, or drops the image that found nothing, does not read
        as correct.
        """
        import tcip_mcp.model_registry as model_registry_mod
        import tcip_mcp.tools.inference_tools as itools
        from tests._binding_fixtures import calibrated_run_fields
        from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

        def _boxes(n: int) -> list[list[float]]:
            return [[10.0 * i, 12.0 * i, 10.0 * i + 24.0, 12.0 * i + 18.0] for i in range(1, n + 1)]

        counts = {"row3_plant07": 1, "row3_plant11": 0, "row9_plant02": 4}
        results = [
            {"image": f"{stem}.jpg", "width": 800, "height": 600, "boxes": _boxes(n),
             "scores": [0.9] * n, "labels": [1] * n, "count": n}
            for stem, n in counts.items()
        ]
        ckpt = tmp_path / "m.pt"
        ckpt.write_bytes(b"x")
        sha = "0f1e2d3c4b5a"
        monkeypatch.setattr(itools, "_run_inference_verified", lambda *a, **kw: {
            "results": results, "image_count": len(results),
            "total_detections": sum(counts.values()), "id_map": None,
            "checkpoint_sha256": sha, "produced_at": "2026-01-01T00:00:00Z",
            **calibrated_run_fields(labels_dir=tmp_path, tiled=False, checkpoint_sha256=sha)})
        monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                            lambda *a, **kw: stub_verified_checkpoint(str(ckpt)))

        out = tmp_path / "dataset" / "predictions" / "baseline" / "2026-01-01"
        res = itools.run_inference(str(ckpt), images_dir=str(tmp_path), output_dir=str(out),
                                   trait="bud_opening")

        assert "error" not in res, res
        assert res["image_count"] == len(counts)
        assert res["output_dir"] == str(out)
        assert sorted(Path(p).stem for p in res["files"]) == sorted(counts)
        written = {p.stem: len(json.loads(p.read_text())["annotations"])
                   for p in out.glob("*.json") if p.name != "operating_point.json"}
        assert written == counts

    def test_deliver_per_image_counts_refuses_a_live_pass_with_no_bucket_naming_each_count(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A live pass with no ``predictions_dir`` has no bucket a reviewer could re-open, and this
        door takes no acknowledgement for the CSV itself, so it always refuses; the refusal still
        names the count measured for each image and the run's own narrowed conf reference,
        distinct from the CSV-facing column, which floors false with nothing on disk behind it.
        """
        import tcip_mcp.model_registry as model_registry_mod
        import tcip_mcp.tools.inference_tools as itools
        from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_HELD_OUT
        from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

        counts = {"row3_plant07.jpg": 2, "row3_plant11.jpg": 0, "row9_plant02.jpg": 17}
        ckpt = tmp_path / "m.pt"
        ckpt.write_bytes(b"x")
        monkeypatch.setattr(itools, "_run_inference_verified", lambda *a, **kw: {
            "results": [{"image": name, "count": n, "scores": [0.9] * n}
                        for name, n in counts.items()],
            "image_count": len(counts), "total_detections": sum(counts.values()),
            "operating_point": {"conf": {"value": 0.6, "validated_against": VALIDATED_HELD_OUT}},
            "validated": True, "conf_source": "calibration"})
        monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint",
                            lambda *a, **kw: stub_verified_checkpoint(str(ckpt)))

        from tests import _operationalization_fixtures as fx

        fx.seed_confirmed_count(tmp_path)
        out_csv = tmp_path / "block_counts.csv"
        # No predictions_dir: nothing on disk backs the count, so the delivery refuses outright.
        res = itools.deliver_per_image_counts(str(ckpt), str(tmp_path), str(out_csv),
                                     trait=fx.COUNT_TRAIT,
                                     calibration_labels_dir=str(tmp_path))

        assert "error" in res
        assert res["image_count"] == len(counts)
        assert res["total_detections"] == sum(counts.values())
        # No bucket, so the CSV-facing column floors false regardless; the run's own narrowed
        # reference travels honestly under its own name instead.
        assert res["operating_point_validated"] == VALIDATED_FALSE
        assert res["run_conf_validated_against"] == VALIDATED_HELD_OUT
        assert not out_csv.exists()


class TestHpoToolOutputSchema:
    def test_run_hyperparameter_search_exists(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "run_hyperparameter_search")
        assert callable(training_tools.run_hyperparameter_search)


# ── Port fallback chain + pytest hermeticity ──────────


def test_post_panel_event_suppressed_under_pytest(monkeypatch):
    """Test runs must never steer a live GUI (PYTEST_CURRENT_TEST is set by pytest itself)."""
    from tcip_mcp.web_client import post_panel_event

    monkeypatch.delenv("TCIP_ALLOW_PANEL_EVENTS", raising=False)
    res = post_panel_event("annotate", "annotate_focus", {"stem": "IMG_X"})
    assert res == {"status": "suppressed_under_pytest", "delivered": False, "url": ""}


def test_post_panel_event_opt_in_bypasses_suppression(monkeypatch):
    from tcip_mcp.web_client import post_panel_event

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    monkeypatch.setenv("TCIP_WEB_PORT", "1")        # nothing listens on port 1
    res = post_panel_event("annotate", "annotate_focus", {})
    assert res["delivered"] is False
    assert res["status"] != "suppressed_under_pytest"   # it really attempted the send


def test_post_panel_event_returns_the_backends_response_body(monkeypatch):
    """``post_panel_event`` reads the real HTTP response rather than discarding it: a
    ``urllib`` round trip against a live backend, since the discarded read this guards is
    specific to that transport, not the ASGI test client the rest of this file uses."""
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
    monkeypatch.setenv("TCIP_WEB_HOST", "127.0.0.1")
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

        res = post_panel_event("app", "active_project_changed", {"name": "no_such_project"})
        assert res["delivered"] is True
        assert res["response"] == {
            "status": "ok", "panel": "app", "event_type": "active_project_changed",
        }
    finally:
        server.should_exit = True
        thread.join(timeout=10)
