"""Integration tests for the push_panel_data HTTP bridge and tool output schemas.

The legacy ``.tcip/events/`` file bridge has been retired. ``push_panel_data``
now POSTs to the tcip-web FastAPI backend; the backend broadcasts to any
subscribed WebSocket clients.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web.app import app


@pytest.fixture
def client():
    return TestClient(app)


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
        for panel in ("annotate", "review", "training", "tuning", "inference", "results"):
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
        # The agent reads gui state back via view_gui_state — a focus event must
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
        from tcip_mcp.web_client import resolve_web_port

        port_file = tmp_path / ".tcip" / "state" / "web_port.txt"
        port_file.parent.mkdir(parents=True)
        port_file.write_text("34567")
        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port(project_root=tmp_path) == 34567

    def test_default_when_neither_available(self, tmp_path: Path, monkeypatch) -> None:
        from tcip_mcp.web_client import DEFAULT_PORT, resolve_web_port

        monkeypatch.delenv("TCIP_WEB_PORT", raising=False)
        assert resolve_web_port(project_root=tmp_path) == DEFAULT_PORT

    def test_host_env_override(self, monkeypatch) -> None:
        from tcip_mcp.web_client import resolve_web_host

        monkeypatch.setenv("TCIP_WEB_HOST", "10.0.0.1")
        assert resolve_web_host() == "10.0.0.1"


# ── Tool output schemas (unchanged from pre-HTTP migration) ─────────────


class TestTrainingToolOutputSchema:
    def test_launch_training_output_has_run_id(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "launch_training")

    def test_check_training_status_schema(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "check_training_status")


class TestInferenceToolOutputSchema:
    def test_run_inference_output_schema(self, data_dir: Path) -> None:
        from tcip_mcp.tools import inference_tools

        assert hasattr(inference_tools, "run_inference")

    def test_export_predictions_schema(self) -> None:
        from tcip_mcp.tools import inference_tools

        assert hasattr(inference_tools, "export_predictions")

    def test_tabulate_counts_schema(self) -> None:
        from tcip_mcp.tools import inference_tools

        assert hasattr(inference_tools, "tabulate_counts")


class TestHpoToolOutputSchema:
    def test_run_hpo_exists(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "run_hpo")
        assert callable(training_tools.run_hpo)


# ── Phase-0 audit fixes: port fallback chain + pytest hermeticity ──────────


def test_resolve_web_port_falls_back_to_repo_root(tmp_path, monkeypatch):
    """After set_active_project repins the platform root to a project, the port file still
    lives under the backend's startup (repo) root — the lookup must find it there instead of
    silently degrading to the default port."""
    from tcip_mcp import web_client

    project = tmp_path / "adopted_project"          # pinned root: no port file here
    project.mkdir()
    repo = tmp_path / "repo"                        # backend's startup root: has the file
    (repo / ".tcip" / "state").mkdir(parents=True)
    (repo / ".tcip" / "state" / "web_port.txt").write_text("23456")

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
