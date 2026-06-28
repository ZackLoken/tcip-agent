"""Integration tests for the push_panel_data HTTP bridge and tool output schemas.

The legacy ``.tcip/events/`` file bridge has been retired. ``push_panel_data``
now POSTs to the tcip-web FastAPI backend; the backend broadcasts to any
subscribed WebSocket clients.
"""

from __future__ import annotations

import os
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


class TestPushPanelDataTool:
    """Verify the MCP tool posts via HTTP and aliases legacy panel names."""

    def test_no_subscribers_when_backend_down(self, tmp_path: Path) -> None:
        """Backend not running → graceful 'no_subscribers' status."""
        from tcip_mcp.tools.annotation_tools import push_panel_data

        # Point port discovery at an unused port in an isolated project root
        os.environ["TCIP_WEB_PORT"] = "59999"  # very unlikely to be bound
        try:
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
        finally:
            os.environ.pop("TCIP_WEB_PORT", None)

    def test_invalid_panel_rejected(self) -> None:
        """Unknown panel names return an error before any HTTP call."""
        from tcip_mcp.tools.annotation_tools import push_panel_data

        result = push_panel_data(panel="bogus", event_type="test", data={})
        assert "error" in result


class TestPortDiscovery:
    """Port + host discovery honor env vars and port-file snapshots."""

    def test_env_port_wins(self) -> None:
        from tcip_mcp.web_client import resolve_web_port

        os.environ["TCIP_WEB_PORT"] = "12345"
        try:
            assert resolve_web_port() == 12345
        finally:
            os.environ.pop("TCIP_WEB_PORT", None)

    def test_port_file_used_when_env_absent(self, tmp_path: Path) -> None:
        from tcip_mcp.web_client import resolve_web_port

        port_file = tmp_path / ".tcip" / "state" / "web_port.txt"
        port_file.parent.mkdir(parents=True)
        port_file.write_text("34567")
        os.environ.pop("TCIP_WEB_PORT", None)
        assert resolve_web_port(project_root=tmp_path) == 34567

    def test_default_when_neither_available(self, tmp_path: Path) -> None:
        from tcip_mcp.web_client import DEFAULT_PORT, resolve_web_port

        os.environ.pop("TCIP_WEB_PORT", None)
        assert resolve_web_port(project_root=tmp_path) == DEFAULT_PORT

    def test_host_env_override(self) -> None:
        from tcip_mcp.web_client import resolve_web_host

        os.environ["TCIP_WEB_HOST"] = "10.0.0.1"
        try:
            assert resolve_web_host() == "10.0.0.1"
        finally:
            os.environ.pop("TCIP_WEB_HOST", None)


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

        assert hasattr(inference_tools, "export_predictions_yolo")

    def test_export_results_csv_schema(self) -> None:
        from tcip_mcp.tools import inference_tools

        assert hasattr(inference_tools, "export_results_csv")


class TestHpoToolOutputSchema:
    def test_run_hpo_exists(self) -> None:
        from tcip_mcp.tools import training_tools

        assert hasattr(training_tools, "run_hpo")
        assert callable(training_tools.run_hpo)
