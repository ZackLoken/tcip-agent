"""Integration test: spawn the real Python MCP server and talk JSON-RPC over stdio.

Validates the full path: subprocess → stdin JSON-RPC → MCPServer → tool call → response.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent


def _send_jsonrpc(proc, obj: dict) -> None:
    """Send a JSON-RPC message as newline-delimited JSON."""
    line = json.dumps(obj) + "\n"
    proc.stdin.write(line.encode("utf-8"))
    proc.stdin.flush()


def _reader_thread(stdout, queue: Queue) -> None:
    """Background thread that reads newline-delimited JSON-RPC messages."""
    while True:
        line = stdout.readline()
        if not line:
            queue.put(None)  # EOF
            return
        line_str = line.decode("utf-8", errors="replace").strip()
        if not line_str:
            continue
        try:
            msg = json.loads(line_str)
            queue.put(msg)
        except json.JSONDecodeError:
            pass


def _recv(queue: Queue, timeout: float = 30.0) -> dict:
    """Receive the next JSON-RPC response, skipping notifications."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = deadline - time.time()
        try:
            msg = queue.get(timeout=min(remaining, 1.0))
        except Empty:
            continue
        if msg is None:
            raise RuntimeError("MCP server closed stdout")
        # Skip notifications (no "id" field) and log messages
        if "id" in msg:
            return msg
    raise TimeoutError("No JSON-RPC response within timeout")


class TestMcpBridge:
    """Spawn the real MCP server and exercise the JSON-RPC protocol."""

    @pytest.fixture(autouse=True)
    def mcp_server(self):
        env = os.environ.copy()

        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tcip_mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(WORKSPACE),
        )

        self.queue: Queue = Queue()
        self.reader = threading.Thread(
            target=_reader_thread, args=(self.proc.stdout, self.queue), daemon=True
        )
        self.reader.start()

        yield

        self.proc.stdin.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def _call(self, method: str, params: dict | None = None, msg_id: int = 1) -> dict:
        msg: dict = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        _send_jsonrpc(self.proc, msg)
        return _recv(self.queue)

    def test_initialize_and_list_tools(self):
        """Server responds to initialize and tools/list."""
        # Initialize
        resp = self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1.0"},
        }, msg_id=1)
        assert "result" in resp, f"Initialize failed: {resp}"
        assert "capabilities" in resp["result"]

        # Send initialized notification (no response expected)
        _send_jsonrpc(self.proc, {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

        # List tools
        resp = self._call("tools/list", {}, msg_id=2)
        assert "result" in resp, f"tools/list failed: {resp}"
        tools = resp["result"].get("tools", [])
        tool_names = {t["name"] for t in tools}
        assert "scan_dataset" in tool_names
        assert len(tool_names) >= 10  # we have 40+ tools


