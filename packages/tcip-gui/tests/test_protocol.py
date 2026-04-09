"""Tests for JSON-RPC protocol message types."""

import json
import pytest

from tcip_gui.protocol import (
    AgentError,
    CanvasAnnotationSaved,
    CanvasBatchComplete,
    CanvasClear,
    CanvasHighlight,
    CanvasLoadImage,
    CanvasReviewComplete,
    CanvasShowPredictions,
    ControlCancel,
    ControlShutdown,
    JsonRpcMessage,
    PermissionRequest,
    PermissionResponse,
    TextDelta,
    TextDone,
    ToolCallResult,
    ToolCallStart,
    TurnComplete,
    UsageStatus,
    UserMessage,
    parse_agent_message,
)


class TestJsonRpcMessage:
    def test_to_line_basic(self):
        msg = JsonRpcMessage(method="test.method", params={"key": "value"}, id="123")
        line = msg.to_line()
        data = json.loads(line)
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "test.method"
        assert data["params"] == {"key": "value"}
        assert data["id"] == "123"

    def test_to_line_no_params(self):
        msg = JsonRpcMessage(method="test.method")
        line = msg.to_line()
        data = json.loads(line)
        assert "params" not in data
        assert "id" not in data

    def test_from_line_roundtrip(self):
        original = JsonRpcMessage(method="foo", params={"x": 1}, id="42")
        line = original.to_line()
        parsed = JsonRpcMessage.from_line(line)
        assert parsed.method == "foo"
        assert parsed.params == {"x": 1}
        assert parsed.id == "42"

    def test_from_line_minimal(self):
        msg = JsonRpcMessage.from_line('{"jsonrpc":"2.0","method":"ping"}')
        assert msg.method == "ping"
        assert msg.params == {}


class TestAgentToGuiMessages:
    def test_text_delta(self):
        td = TextDelta(text="hello ")
        rpc = td.to_rpc()
        assert rpc.method == "assistant.text_delta"
        assert rpc.params["text"] == "hello "

    def test_text_done(self):
        td = TextDone(text="full response")
        rpc = td.to_rpc()
        assert rpc.method == "assistant.text_done"

    def test_tool_call_start(self):
        tcs = ToolCallStart(tool_id="t1", name="read_file", input={"path": "/a"})
        rpc = tcs.to_rpc()
        assert rpc.method == "tool.call_start"
        assert rpc.params["name"] == "read_file"

    def test_tool_call_result(self):
        tcr = ToolCallResult(tool_id="t1", output="contents", is_error=False)
        rpc = tcr.to_rpc()
        assert rpc.params["is_error"] is False

    def test_permission_request(self):
        pr = PermissionRequest(
            request_id="p1", tool="launch_training",
            input={"lr": 0.001}, description="Start training", level="FullAccess",
        )
        rpc = pr.to_rpc()
        assert rpc.method == "permission.request"
        assert rpc.params["id"] == "p1"

    def test_usage_status(self):
        u = UsageStatus(input_tokens=100, output_tokens=50, cost=0.002)
        rpc = u.to_rpc()
        assert rpc.params["cost"] == 0.002

    def test_turn_complete(self):
        tc = TurnComplete(turn_number=3)
        rpc = tc.to_rpc()
        assert rpc.params["turn_number"] == 3

    def test_agent_error(self):
        e = AgentError(code=-32000, message="runtime error")
        rpc = e.to_rpc()
        assert rpc.method == "error"
        assert rpc.params["code"] == -32000


class TestGuiToAgentMessages:
    def test_user_message(self):
        um = UserMessage(text="Hello agent")
        rpc = um.to_rpc()
        assert rpc.method == "user.message"
        assert rpc.id is not None  # UUID generated

    def test_permission_response_approve(self):
        pr = PermissionResponse(request_id="p1", allowed=True)
        rpc = pr.to_rpc()
        assert rpc.params["allowed"] is True
        assert "reason" not in rpc.params

    def test_permission_response_deny(self):
        pr = PermissionResponse(request_id="p1", allowed=False, reason="Too expensive")
        rpc = pr.to_rpc()
        assert rpc.params["reason"] == "Too expensive"

    def test_control_cancel(self):
        cc = ControlCancel()
        assert cc.to_rpc().method == "control.cancel"

    def test_control_shutdown(self):
        cs = ControlShutdown()
        assert cs.to_rpc().method == "control.shutdown"


class TestParseAgentMessage:
    def test_parse_text_delta(self):
        msg = JsonRpcMessage(method="assistant.text_delta", params={"text": "hi"})
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, TextDelta)
        assert parsed.text == "hi"

    def test_parse_tool_call_start(self):
        msg = JsonRpcMessage(
            method="tool.call_start",
            params={"id": "t1", "name": "grep_search", "input": {"query": "test"}},
        )
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, ToolCallStart)
        assert parsed.name == "grep_search"

    def test_parse_unknown_returns_raw(self):
        msg = JsonRpcMessage(method="unknown.method", params={"x": 1})
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, JsonRpcMessage)

    def test_parse_error(self):
        msg = JsonRpcMessage(method="error", params={"code": -1, "message": "boom"})
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, AgentError)
        assert parsed.message == "boom"

    def test_parse_usage(self):
        msg = JsonRpcMessage(
            method="status.usage",
            params={"input_tokens": 500, "output_tokens": 200, "cost": 0.01},
        )
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, UsageStatus)
        assert parsed.input_tokens == 500


class TestCanvasProtocol:
    def test_canvas_load_image(self):
        msg = CanvasLoadImage(path="/img.jpg", annotations=[{"cls": 0}])
        rpc = msg.to_rpc()
        assert rpc.method == "canvas.load_image"
        assert rpc.params["path"] == "/img.jpg"
        assert rpc.params["annotations"] == [{"cls": 0}]

    def test_canvas_show_predictions(self):
        msg = CanvasShowPredictions(predictions_path="/preds.txt")
        rpc = msg.to_rpc()
        assert rpc.method == "canvas.show_predictions"
        assert rpc.params["predictions_path"] == "/preds.txt"

    def test_canvas_clear(self):
        msg = CanvasClear()
        assert msg.to_rpc().method == "canvas.clear"

    def test_canvas_highlight(self):
        msg = CanvasHighlight(annotation_ids=["0", "1", "3"])
        rpc = msg.to_rpc()
        assert rpc.method == "canvas.highlight"
        assert rpc.params["annotation_ids"] == ["0", "1", "3"]

    def test_canvas_annotation_saved(self):
        msg = CanvasAnnotationSaved(image_path="/img.jpg", count=5)
        rpc = msg.to_rpc()
        assert rpc.method == "canvas.annotation_saved"
        assert rpc.params["count"] == 5

    def test_canvas_review_complete(self):
        msg = CanvasReviewComplete(image_path="/img.jpg", accepted=10, edited=3, rejected=2)
        rpc = msg.to_rpc()
        assert rpc.method == "canvas.review_complete"
        assert rpc.params["accepted"] == 10

    def test_canvas_batch_complete(self):
        msg = CanvasBatchComplete(summary={"total": 50, "annotated": 30})
        rpc = msg.to_rpc()
        assert rpc.method == "canvas.batch_complete"
        assert rpc.params["summary"]["total"] == 50

    def test_parse_canvas_load_image(self):
        msg = JsonRpcMessage(method="canvas.load_image", params={"path": "/img.jpg"})
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, CanvasLoadImage)
        assert parsed.path == "/img.jpg"

    def test_parse_canvas_show_predictions(self):
        msg = JsonRpcMessage(method="canvas.show_predictions", params={"predictions_path": "/p.txt"})
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, CanvasShowPredictions)

    def test_parse_canvas_clear(self):
        msg = JsonRpcMessage(method="canvas.clear", params={})
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, CanvasClear)

    def test_parse_canvas_highlight(self):
        msg = JsonRpcMessage(method="canvas.highlight", params={"annotation_ids": ["0"]})
        parsed = parse_agent_message(msg)
        assert isinstance(parsed, CanvasHighlight)
        assert parsed.annotation_ids == ["0"]
