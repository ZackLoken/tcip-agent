"""Tests for the AgentBridge — JSON-RPC parsing, message routing, process lifecycle."""

import json
import pytest
from unittest.mock import MagicMock, patch

from tcip_gui.protocol import (
    JsonRpcMessage,
    TextDelta,
    TextDone,
    ToolCallStart,
    ToolCallResult,
    UsageStatus,
    AgentError,
    UserMessage,
    ControlShutdown,
    CanvasLoadImage,
    CanvasShowPredictions,
    CanvasClear,
    CanvasHighlight,
)
from tcip_gui.bridge import AgentBridge


class TestAgentBridgeInit:
    def test_bridge_creation(self):
        bridge = AgentBridge("echo", ["hello"])
        assert bridge._command == "echo"
        assert bridge._args == ["hello"]
        assert bridge._process is None
        assert not bridge.is_running()


class TestMessageDispatch:
    """Test that _dispatch_message routes to correct signals."""

    def test_dispatch_text_delta(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.text_delta.connect(lambda t: received.append(("text_delta", t)))

        bridge._dispatch_message(TextDelta(text="hello"))
        assert received == [("text_delta", "hello")]

    def test_dispatch_text_done(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.text_done.connect(lambda t: received.append(t))

        bridge._dispatch_message(TextDone(text="complete"))
        assert received == ["complete"]

    def test_dispatch_tool_call_start(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.tool_call_start.connect(lambda id, name, inp: received.append((id, name)))

        bridge._dispatch_message(ToolCallStart(tool_id="t1", name="read_file", input={"path": "/a"}))
        assert received == [("t1", "read_file")]

    def test_dispatch_tool_call_result(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.tool_call_result.connect(lambda id, out, err: received.append((id, err)))

        bridge._dispatch_message(ToolCallResult(tool_id="t1", output="data", is_error=False))
        assert received == [("t1", False)]

    def test_dispatch_usage_update(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.usage_update.connect(lambda i, o, c: received.append((i, o, c)))

        bridge._dispatch_message(UsageStatus(input_tokens=100, output_tokens=50, cost=0.002))
        assert received == [(100, 50, 0.002)]

    def test_dispatch_error(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.agent_error.connect(lambda code, msg: received.append((code, msg)))

        bridge._dispatch_message(AgentError(code=-1, message="fail"))
        assert received == [(-1, "fail")]

    def test_dispatch_unknown_type(self):
        """Unknown types should not crash."""
        bridge = AgentBridge("echo", [])
        bridge._dispatch_message("something unexpected")  # Should not raise


class TestMessageSerialization:
    def test_user_message_serialization(self):
        um = UserMessage(text="test")
        line = um.to_rpc().to_line()
        data = json.loads(line)
        assert data["method"] == "user.message"
        assert data["params"]["text"] == "test"

    def test_shutdown_serialization(self):
        cs = ControlShutdown()
        line = cs.to_rpc().to_line()
        data = json.loads(line)
        assert data["method"] == "control.shutdown"


class TestCanvasDispatch:
    """Test that canvas messages are dispatched to canvas signals."""

    def test_dispatch_canvas_load_image(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.canvas_load_image.connect(lambda p, a: received.append(("load", p)))
        bridge._dispatch_message(CanvasLoadImage(path="/img.jpg", annotations=None))
        assert received == [("load", "/img.jpg")]

    def test_dispatch_canvas_show_predictions(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.canvas_show_predictions.connect(lambda p: received.append(p))
        bridge._dispatch_message(CanvasShowPredictions(predictions_path="/preds.txt"))
        assert received == ["/preds.txt"]

    def test_dispatch_canvas_clear(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.canvas_clear.connect(lambda: received.append("clear"))
        bridge._dispatch_message(CanvasClear())
        assert received == ["clear"]

    def test_dispatch_canvas_highlight(self):
        bridge = AgentBridge("echo", [])
        received = []
        bridge.canvas_highlight.connect(lambda ids: received.append(ids))
        bridge._dispatch_message(CanvasHighlight(annotation_ids=["0", "1"]))
        assert received == [["0", "1"]]
