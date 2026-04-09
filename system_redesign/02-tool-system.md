# 02 — Tool System

## Three-Layer Tool Architecture

Claw-code has three sources of tools:

1. **Built-in** (40 tool specs in `tools/src/lib.rs`) — statically defined
2. **Plugin** — external tools from plugin registry
3. **Runtime** — dynamically registered (MCP, LSP, custom)

## GlobalToolRegistry

```rust
pub struct GlobalToolRegistry {
    plugin_tools: Vec<PluginTool>,
    runtime_tools: Vec<RuntimeToolDefinition>,
    enforcer: Option<PermissionEnforcer>,
}
```

Dispatches tool calls by name, checks permissions, deserializes input, calls handler,
serializes output.

## Tool Metadata (ToolSpec)

```rust
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,          // JSON Schema
    pub required_permission: PermissionMode,
}
```

Each tool declares its permission level upfront:
- `ReadOnly` — file reads, searches, web fetch
- `WorkspaceWrite` — file writes, task updates
- `DangerFullAccess` — bash, workers, agents, MCP

## Complete Tool Catalog (40 tools)

### File I/O (6)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `bash` | DangerFullAccess | Shell command execution with sandboxing |
| `read_file` | ReadOnly | Line-windowed file reading (max 10MB) |
| `write_file` | WorkspaceWrite | Full file replacement (max 10MB) |
| `edit_file` | WorkspaceWrite | Targeted string replacement |
| `glob_search` | ReadOnly | Filename pattern matching |
| `grep_search` | ReadOnly | Regex search with context lines |

### Web (2)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `WebFetch` | ReadOnly | Fetch webpage content |
| `WebSearch` | ReadOnly | Search the web |

### Tasks / Sub-agents (7)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `TaskCreate` | DangerFullAccess | Spawn a sub-task |
| `TaskGet` | ReadOnly | Check task status |
| `TaskList` | ReadOnly | List all tasks |
| `TaskStop` | WorkspaceWrite | Stop a running task |
| `TaskUpdate` | WorkspaceWrite | Send message to task |
| `TaskOutput` | ReadOnly | Get task output |
| `RunTaskPacket` | DangerFullAccess | Execute structured task packet |

### Workers (8)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `WorkerCreate` | DangerFullAccess | Spawn a worker process |
| `WorkerGet` | ReadOnly | Check worker status |
| `WorkerObserve` | ReadOnly | Watch worker events |
| `WorkerResolveTrust` | WorkspaceWrite | Approve trust prompt |
| `WorkerAwaitReady` | ReadOnly | Wait for worker ready |
| `WorkerSendPrompt` | DangerFullAccess | Send prompt to worker |
| `WorkerRestart` | DangerFullAccess | Restart a worker |
| `WorkerTerminate` | DangerFullAccess | Kill a worker |

### Teams & Cron (5)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `TeamCreate` | DangerFullAccess | Create worker team |
| `TeamDelete` | DangerFullAccess | Delete team |
| `CronCreate` | DangerFullAccess | Schedule recurring task |
| `CronDelete` | DangerFullAccess | Remove scheduled task |
| `CronList` | ReadOnly | List cron entries |

### Code Intelligence (4)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `LSP` | ReadOnly | Language server queries (symbols, refs, diagnostics) |
| `NotebookEdit` | WorkspaceWrite | Edit Jupyter notebooks |
| `REPL` | DangerFullAccess | Interactive REPL |
| `PowerShell` | DangerFullAccess | PowerShell execution |

### MCP (5)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `MCP` | DangerFullAccess | Invoke an MCP server tool |
| `ListMcpResources` | ReadOnly | List MCP server resources |
| `ReadMcpResource` | ReadOnly | Read MCP resource by URI |
| `McpAuth` | WorkspaceWrite | MCP OAuth authentication |
| `RemoteTrigger` | DangerFullAccess | Trigger remote MCP action |

### User Interaction (3)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `SendUserMessage` | ReadOnly | Display message to user |
| `AskUserQuestion` | ReadOnly | Prompt user for input |
| `TodoWrite` | WorkspaceWrite | Manage todo list |

### Meta (8)
| Tool | Permission | Purpose |
|------|-----------|---------|
| `Skill` | ReadOnly | Execute a named skill |
| `Agent` | DangerFullAccess | Spawn a sub-agent |
| `ToolSearch` | ReadOnly | Search available tools |
| `Config` | ReadOnly | Read configuration |
| `EnterPlanMode` | ReadOnly | Switch to planning mode |
| `ExitPlanMode` | ReadOnly | Switch back to execution |
| `StructuredOutput` | ReadOnly | Emit structured JSON |
| `Sleep` | ReadOnly | Wait (for polling scenarios) |

## Execute Dispatch Pattern

```rust
fn execute_tool_with_enforcer(
    enforcer: Option<&PermissionEnforcer>,
    name: &str,
    input: &Value,
) -> Result<String, String>
```

1. Optional permission check (if enforcer present)
2. Deserialize `Value` → typed input struct
3. Call handler function
4. Serialize result → JSON string

## Bash Tool Detail

The bash tool is the most complex native tool:

```rust
pub struct BashCommandInput {
    command: String,
    timeout: Option<u64>,               // Milliseconds
    run_in_background: Option<bool>,
    dangerously_disable_sandbox: Option<bool>,
    namespace_restrictions: Option<bool>,
    isolate_network: Option<bool>,
    filesystem_mode: Option<FilesystemIsolationMode>,
    allowed_mounts: Option<Vec<String>>,
}
```

Execution flow:
1. Get sandbox config from environment
2. If `run_in_background`: spawn detached, return task ID
3. Otherwise: async execution with timeout via tokio
4. Build command (bubblewrap sandbox on Linux, or raw `sh -lc`)
5. Capture stdout/stderr, truncate if > 10MB
6. Return status code + output

### Bash Validation (pre-execution)

```rust
pub enum ValidationResult {
    Allow,
    Block { reason: String },
    Warn { message: String },
}
```

Checks:
- **Read-only mode**: blocks write commands (cp, mv, rm, mkdir, chmod, etc.)
- **Destructive patterns**: warns on `rm -rf /`, fork bombs, `dd` direct writes
- **Command intent classification**: Read, Write, Destructive, Network, ProcessManagement

## File Operations Detail

- `read_file(path, offset, limit)` — line-windowed, max 10MB, binary detection (NUL bytes)
- `write_file(path, content)` — full replacement, max 10MB, workspace boundary check
- `edit_file(path, old, new)` — targeted string replacement
- `glob_search(pattern)` — filename matching
- `grep_search(pattern, path, options)` — regex with context (-B, -A, -C), case insensitive
