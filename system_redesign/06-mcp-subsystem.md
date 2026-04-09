# 06 — MCP Subsystem

The MCP (Model Context Protocol) layer spans 5 files in the runtime crate.
This is how claw-code extends itself beyond its 6 native tools.

## Architecture

```
ConfigLoader → McpServerConfig (stdio/sse/http/ws/sdk/proxy)
    → McpClientBootstrap (transport + naming)
        → McpToolRegistry (connection tracking + tool/resource inventory)
            → McpLifecycleState (phase tracking + degraded mode)
```

## Server Configuration

```rust
pub enum McpServerConfig {
    Stdio(McpStdioServerConfig),           // Local process (command + args + env)
    Sse(McpRemoteServerConfig),            // SSE endpoint
    Http(McpRemoteServerConfig),           // HTTP endpoint
    Ws(McpWebSocketServerConfig),          // WebSocket
    Sdk(McpSdkServerConfig),              // SDK name
    ManagedProxy(McpManagedProxyServerConfig),  // Anthropic CCR proxy
}
```

Stdio config (most relevant for our use case):
```rust
pub struct McpStdioServerConfig {
    command: String,
    args: Vec<String>,
    env: HashMap<String, String>,
    tool_call_timeout_ms: Option<u64>,
}
```

## Tool Naming

MCP tools are prefixed to avoid collisions with native tools:
- `mcp_tool_name(server, tool)` → `"mcp__server__tool"`
- `mcp_tool_prefix(server)` → `"mcp__server__"`
- Non-alphanumeric chars replaced with `_`

## JSON-RPC Wire Protocol

```rust
pub struct JsonRpcRequest<T> {
    jsonrpc: "2.0",
    id: JsonRpcId,
    method: String,
    params: Option<T>,
}
```

MCP operations:
- `initialize` — handshake, get server info + capabilities
- `tools/list` — discover available tools
- `tools/call` — invoke tool with arguments
- `resources/list` — discover resources
- `resources/read` — read resource by URI

### Timeouts
- Initialize: 10s
- ListTools: 30s
- ToolCall: per-server config, default 60s

## McpToolRegistry

```rust
pub struct McpToolRegistry {
    inner: Arc<Mutex<HashMap<String, McpServerState>>>,
    manager: Arc<OnceLock<Arc<Mutex<McpServerManager>>>>,
}

pub struct McpServerState {
    server_name: String,
    status: McpConnectionStatus,
    tools: Vec<McpToolInfo>,
    resources: Vec<McpResourceInfo>,
}
```

## Lifecycle State Machine

```rust
pub enum McpLifecyclePhase {
    ConfigLoad,
    ServerRegistration,
    SpawnConnect,
    InitializeHandshake,
    ToolDiscovery,
    ResourceDiscovery,
    Ready,
    Invocation,
    ErrorSurfacing,
    Shutdown,
    Cleanup,
}
```

## Degraded Mode

When some MCP servers fail to start but others succeed:

```rust
pub struct McpDegradedReport {
    working_servers: Vec<String>,
    failed_servers: Vec<String>,
    available_tools: Vec<String>,
    missing_tools: Vec<String>,
}
```

Runtime continues with partial service. The model is told which tools are available
and which servers failed. This is a critical design decision — partial failure
doesn't crash the agent.

## Error Surface

```rust
pub struct McpErrorSurface {
    phase: McpLifecyclePhase,
    server_name: Option<String>,
    message: String,
    context: BTreeMap<String, String>,
    recoverable: bool,
    timestamp: u64,
}
```

The `recoverable` flag allows the system to decide whether to retry or report.
