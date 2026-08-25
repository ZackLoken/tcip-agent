"""Integration tests for the push_panel_data HTTP bridge and tool output schemas.

The legacy ``.tcip/events/`` file bridge has been retired. ``push_panel_data``
now POSTs to the tcip-web FastAPI backend; the backend broadcasts to any
subscribed WebSocket clients.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.web_client import VALID_PANELS
from tcip_web.app import app

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src"


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

    def test_recent_events_returned(self, client: TestClient) -> None:
        client.post(
            "/api/events/tuning",
            json={"event_type": "trial_update", "data": {"trial": 1}},
        )
        client.post(
            "/api/events/tuning",
            json={"event_type": "trial_update", "data": {"trial": 2}},
        )
        resp = client.get("/api/events/tuning/recent?limit=2")
        events = resp.json()["events"]
        assert len(events) == 2
        assert events[-1]["data"]["trial"] == 2

    def test_review_focus_persists_advisory_state(self, client: TestClient) -> None:
        # The agent reads gui state back via view_gui_state: a focus event must
        # land there even though the browser applies it with local setters only.
        resp = client.post(
            "/api/events/app",
            json={
                "event_type": "review_focus",
                "data": {
                    "subject": "catkin",
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
        state = client.get("/api/dataset/state").json()
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
                "data": {"subject": "bush", "date": "2-11-26", "mode": "polygon", "active_subject": "catkin"},
            },
        )
        assert resp.status_code == 200
        state = client.get("/api/dataset/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "polygon"
        assert state["active_subject"] == "catkin"

    def test_the_focus_tools_own_annotate_event_reaches_the_advisory_state(
        self, client: TestClient, data_dir: Path, monkeypatch,
    ) -> None:
        """The state an agent reads back is driven by the event the focus tool really posts.

        The payload is taken from the producer rather than written here, so the two halves of the
        bridge are held to the same key names: a producer and a consumer that stop agreeing on
        what a field is called cannot both keep passing.
        """
        from tcip_mcp import web_client
        from tcip_mcp.tools.annotation_tools import focus

        posted: dict = {}

        def _capture(panel: str, event_type: str, data: dict) -> dict:
            posted.update(panel=panel, event_type=event_type, data=data)
            return {"delivered": True, "status": "ok"}

        monkeypatch.setattr(web_client, "post_panel_event", _capture)
        res = focus("annotate", str(data_dir), str(data_dir), "catkin", "2-11-26",
                    mode="point", image_index=2)
        assert "error" not in res, res
        assert posted["event_type"] == "annotate_focus"

        resp = client.post(f"/api/events/{posted['panel']}",
                           json={"event_type": posted["event_type"], "data": posted["data"]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        state = client.get("/api/dataset/state").json()
        assert state["active_tab"] == "annotate"
        assert state["mode"] == "point"
        assert state["active_subject"] == "catkin"

    def test_the_browser_tabs_and_the_backend_agree_on_the_panel_names(self) -> None:
        """Every GUI tab subscribes to the panel of its own name.

        The vocabulary is written down in three places, one Python set and two TypeScript
        literals, so a panel renamed or dropped in one of them leaves either a tab with no live
        subscription or a subscription no tab is listening on. ``app`` is the app-level channel
        and has no tab of its own.
        """
        app_tsx = (FRONTEND_SRC / "App.tsx").read_text(encoding="utf-8")
        types_ts = (FRONTEND_SRC / "store" / "types.ts").read_text(encoding="utf-8")

        subscribed = re.search(r"const TAB_PANELS: TabName\[\] = \[(.*?)\];", app_tsx, re.S)
        declared = re.search(r"export type TabName =(.*?);", types_ts, re.S)
        assert subscribed is not None, "TAB_PANELS is no longer where this test reads it"
        assert declared is not None, "TabName is no longer where this test reads it"

        subscribed_names = set(re.findall(r'"([^"]+)"', subscribed.group(1)))
        declared_names = set(re.findall(r'"([^"]+)"', declared.group(1)))
        assert subscribed_names, "no panel names parsed out of TAB_PANELS"
        assert subscribed_names == declared_names
        assert subscribed_names == set(VALID_PANELS) - {"app"}


class TestPushPanelDataTool:
    """Verify the MCP tool posts via HTTP and aliases legacy panel names."""

    def test_no_subscribers_when_backend_down(self, tmp_path: Path, monkeypatch) -> None:
        """Backend not running → graceful 'no_subscribers' status."""
        from tcip_mcp.tools.annotation_tools import push_panel_data

        # Point port discovery at an unused port in an isolated project root
        monkeypatch.setenv("TCIP_WEB_PORT", "59999")  # very unlikely to be bound
        result = push_panel_data(
            panel="training",
            event_type="metrics_update",
            data={"epoch": 1},
        )
        # Either the connection was refused (no_subscribers) or a URL error;
        # both are acceptable. Tool must not raise.
        assert "status" in result or "error" in result
        # Panel name preserved in result
        assert result.get("panel") == "training"

    def test_invalid_panel_rejected(self) -> None:
        """Unknown panel names return an error before any HTTP call."""
        from tcip_mcp.tools.annotation_tools import push_panel_data

        result = push_panel_data(panel="bogus", event_type="test", data={})
        assert "error" in result


class TestPortDiscovery:
    """Port + host discovery honor env vars and port-file snapshots."""

    def test_env_port_wins(self, monkeypatch) -> None:
        from tcip_mcp.web_client import resolve_web_port

        monkeypatch.setenv("TCIP_WEB_PORT", "12345")
        assert resolve_web_port() == 12345

    def test_port_file_used_when_env_absent(self, tmp_path: Path, monkeypatch) -> None:
        import tcip_store as ts
        from tcip_mcp.web_client import backend_port_key, resolve_web_port

        ts.replace(backend_port_key(root=tmp_path), "34567")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port(project_root=tmp_path) == 34567

    def test_the_port_the_backend_writes_is_the_port_a_tool_process_reads(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The backend's own writer and the MCP-side resolver meet at one file.

        Both halves run here rather than the file being hand-written, so a tool in another process
        finds the port the backend actually bound instead of quietly falling back to the default.
        """
        from tcip_mcp.web_client import DEFAULT_PORT, resolve_web_port
        from tcip_web.__main__ import _write_port_file

        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
        bound = 41871
        assert bound != DEFAULT_PORT
        _write_port_file(bound)
        assert resolve_web_port(project_root=tmp_path) == bound
        assert resolve_web_port() == bound

    def test_default_when_neither_available(self, tmp_path: Path, monkeypatch) -> None:
        from tcip_mcp.web_client import DEFAULT_PORT, resolve_web_port

        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port(project_root=tmp_path) == DEFAULT_PORT

    def test_an_unreadable_recorded_port_falls_through_to_the_default(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A handoff nothing can turn into a port is not a port.

        This lookup runs before the backend is known to be up, so it reports the default and
        lets the caller try rather than raising on a record it cannot use.
        """
        import tcip_store as ts
        from tcip_mcp.web_client import DEFAULT_PORT, backend_port_key, resolve_web_port

        ts.replace(backend_port_key(root=tmp_path), "not a port")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port(project_root=tmp_path) == DEFAULT_PORT

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
        from tcip_mcp.pipelines.training.generic_trainer import get_run
        from tcip_mcp.tools import training_tools

        images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
        val_images, val_labels = tmp_path / "val_images", tmp_path / "val_labels"
        for d in (images_dir, labels_dir, val_images, val_labels):
            d.mkdir()
        for i in range(2):
            Image.new("RGB", (128, 128)).save(images_dir / f"t{i}.png")
            json_io.write_annotations(
                str(labels_dir / f"t{i}.json"),
                [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 128, 128)
        Image.new("RGB", (128, 128)).save(val_images / "v0.png")
        json_io.write_annotations(
            str(val_labels / "v0.json"),
            [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 128, 128)

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
                     "subject": "catkin", "val_images_dir": str(val_images),
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

    def test_check_training_status_answers_for_the_run_it_was_asked_about(
        self, tmp_path: Path,
    ) -> None:
        """A status read carries the identifier of the run it describes, plus that run's own
        progress, so a caller tracking several runs at once can tell the answers apart. An
        identifier that names no run is refused rather than answered for some other run.
        """
        from tcip_mcp.pipelines.training.generic_trainer import create_run
        from tcip_mcp.tools import training_tools

        early = create_run({"seed": 11}, str(tmp_path / "early"))
        early.status, early.current_epoch, early.best_metric = "running", 1, 0.81
        late = create_run({"seed": 12}, str(tmp_path / "late"))
        late.status, late.current_epoch, late.best_metric = "completed", 9, 0.07

        status = training_tools.check_training_status(late.run_id)
        assert status["run_id"] == late.run_id
        assert status["run_id"] != early.run_id
        assert status["status"] == "completed"
        assert status["epoch"] == 9
        assert status["best_metric"] == 0.07
        assert status["output_dir"] == str(tmp_path / "late")

        assert "error" in training_tools.check_training_status("run_that_was_never_created")


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
        res = run_inference(str(ckpt), images_dir=str(tmp_path), dry_run=True, tile=True,
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

    def test_export_predictions_writes_one_file_per_image_carrying_that_images_detections(
        self, tmp_path: Path, monkeypatch, seed_catkin_trait_spec,
    ) -> None:
        """A prediction bucket holds one file per image the pass saw, named for that image's own
        stem and holding that image's own detections.

        The three images here have three different detection counts, one of them zero, so a bucket
        that pairs a file to the wrong image, or drops the image that found nothing, does not read
        as correct.
        """
        import tcip_mcp.tools.inference_tools as itools
        from tests._binding_fixtures import calibrated_run_fields

        def _boxes(n: int) -> list[list[float]]:
            return [[10.0 * i, 12.0 * i, 10.0 * i + 24.0, 12.0 * i + 18.0] for i in range(1, n + 1)]

        counts = {"row3_plant07": 1, "row3_plant11": 0, "row9_plant02": 4}
        results = [
            {"image": f"{stem}.jpg", "width": 800, "height": 600, "boxes": _boxes(n),
             "scores": [0.9] * n, "labels": [1] * n, "count": n}
            for stem, n in counts.items()
        ]
        monkeypatch.setattr(itools, "run_inference", lambda **kw: {
            "results": results, "image_count": len(results),
            "total_detections": sum(counts.values()), "id_map": None,
            "checkpoint_sha256": "0f1e2d3c4b5a", "produced_at": "2026-01-01T00:00:00Z",
            **calibrated_run_fields(labels_dir=tmp_path, tiled=False)})

        out = tmp_path / "dataset" / "predictions" / "baseline" / "2026-01-01"
        res = itools.export_predictions("m.pt", images_dir=str(tmp_path), output_dir=str(out),
                                        trait="catkin")

        assert "error" not in res, res
        assert res["image_count"] == len(counts)
        assert res["output_dir"] == str(out)
        assert sorted(Path(p).stem for p in res["files"]) == sorted(counts)
        written = {p.stem: len(json.loads(p.read_text())["annotations"])
                   for p in out.glob("*.json") if p.name != "operating_point.json"}
        assert written == counts

    def test_tabulate_counts_writes_each_images_own_count_into_the_csv(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The delivered CSV carries the count measured for each image, one row each.

        The counts across the block are deliberately uneven, including a zero, so a CSV that
        flattens them or drops the empty frame reads differently from one that reports what was
        measured.
        """
        import csv

        import tcip_mcp.tools.inference_tools as itools
        from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT

        counts = {"row3_plant07.jpg": 2, "row3_plant11.jpg": 0, "row9_plant02.jpg": 17}
        monkeypatch.setattr(itools, "run_inference", lambda **kw: {
            "results": [{"image": name, "count": n, "scores": [0.9] * n}
                        for name, n in counts.items()],
            "image_count": len(counts), "total_detections": sum(counts.values()),
            "operating_point": {"conf": {"value": 0.6, "validated_against": VALIDATED_HELD_OUT}},
            "validated": True, "conf_source": "calibration"})

        from tests import _operationalization_fixtures as fx

        fx.seed_confirmed_count(tmp_path)
        out_csv = tmp_path / "block_counts.csv"
        # No predictions_dir: the CSV is the provisional one but still carries each measured count.
        res = itools.tabulate_counts("m.pt", str(tmp_path), str(out_csv),
                                     trait=fx.COUNT_TRAIT,
                                     calibration_labels_dir=str(tmp_path),
                                     acknowledge_unvalidated=True)

        assert "error" not in res, res
        assert res["csv_path"] == str(out_csv)
        assert res["image_count"] == len(counts)
        assert res["total_detections"] == sum(counts.values())
        assert res["operating_point_validated"] == VALIDATED_HELD_OUT
        rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
        assert {r["image"]: int(r["detection_count"]) for r in rows} == counts


class TestHpoToolOutputSchema:
    def test_run_hpo_exists(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "run_hpo")
        assert callable(training_tools.run_hpo)


# ── Port fallback chain + pytest hermeticity ──────────


def test_resolve_web_port_falls_back_to_repo_root(tmp_path, monkeypatch):
    """After set_active_project repins the platform root to a project, the port file still
    lives under the backend's startup (repo) root: the lookup must find it there instead of
    silently degrading to the default port."""
    import tcip_store as ts
    from tcip_mcp import web_client

    project = tmp_path / "adopted_project"          # pinned root: no port record here
    project.mkdir()
    repo = tmp_path / "repo"                        # backend's startup root: has the record
    repo.mkdir()
    ts.replace(web_client.backend_port_key(root=repo), "23456")

    monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(project))
    monkeypatch.setattr(web_client, "_repo_root", lambda: repo)
    assert web_client.resolve_web_port() == 23456


def test_post_panel_event_suppressed_under_pytest(monkeypatch):
    """Test runs must never steer a live GUI (PYTEST_CURRENT_TEST is set by pytest itself)."""
    from tcip_mcp.web_client import post_panel_event

    monkeypatch.delenv("TCIP_ALLOW_PANEL_EVENTS", raising=False)
    res = post_panel_event("annotate", "focus", {"stem": "IMG_X"})
    assert res == {"status": "suppressed_under_pytest", "delivered": False, "url": ""}


def test_post_panel_event_opt_in_bypasses_suppression(monkeypatch):
    from tcip_mcp.web_client import post_panel_event

    monkeypatch.setenv("TCIP_ALLOW_PANEL_EVENTS", "1")
    monkeypatch.setenv("TCIP_WEB_PORT", "1")        # nothing listens on port 1
    res = post_panel_event("annotate", "focus", {})
    assert res["delivered"] is False
    assert res["status"] != "suppressed_under_pytest"   # it really attempted the send
