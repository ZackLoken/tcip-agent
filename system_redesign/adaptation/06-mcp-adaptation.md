# 06 — MCP Adaptation

## What claw-code has

Full MCP lifecycle: config → spawn → handshake → tool discovery → invocation.
6 transport types (stdio, SSE, HTTP, WebSocket, SDK, ManagedProxy).
Degraded mode when some servers fail. Tool naming with server prefixes.

## What carries over as-is

- **Stdio transport** — we spawn a Python MCP server as a child process. This is
  exactly claw-code's `McpStdioServerConfig { command, args, env, timeout }`.
- **JSON-RPC wire protocol** — standard MCP. Keep as-is.
- **Tool discovery** — `tools/list` at startup. Keep.
- **Tool naming** — `mcp__pipeline__launch_training`. Keep the prefix convention.
- **Degraded mode** — if the Python server crashes mid-session, the agent should
  know which tools are unavailable and adapt. Keep this.
- **McpLifecyclePhase state machine** — proper lifecycle tracking. Keep.

## What we simplify

### Single MCP server for MVP
Claw-code supports N servers with different transports. We have one:

```json
{
  "pipeline_server": {
    "type": "stdio",
    "command": "python",
    "args": ["-m", "tcip_mcp"],
    "env": { "CUDA_VISIBLE_DEVICES": "0" },
    "tool_call_timeout_ms": 300000
  }
}
```

5-minute timeout because training validation and inference can be slow.

### Drop remote transports for MVP
No SSE, HTTP, WebSocket, SDK, ManagedProxy needed. Just stdio. Can add remote
support later if we move to client-server deployment.

## What's new

### Long-running tool calls
MCP tools/call has a timeout. Training runs for hours. Two approaches:

**Option A**: Training tool returns immediately with a run_id. Agent polls via
`check_training_status(run_id)`. This is claw-code's background task pattern.

**Option B**: MCP server sends progress notifications during long operations.
MCP spec supports notifications but claw-code doesn't heavily use them.

**Decision**: Option A. Matches claw-code's existing pattern. Simpler. The GUI
gets training progress by the agent periodically calling `check_training_status`,
or the GUI directly reads TensorBoard logs independently of the agent.

### MCP server lifecycle — sibling process model
The GUI process (PyQt6) spawns both the Rust agent and Python MCP server as sibling
child processes. The GUI tells the agent where to find the MCP server via config
(stdio pipe or socket). This is robust: if the agent crashes, the MCP server (and any
running training) survives. When the app closes, both children terminate.

The agent connects to the already-running MCP server rather than spawning it.
This is the only deviation from claw-code's default "agent spawns MCP" pattern.
