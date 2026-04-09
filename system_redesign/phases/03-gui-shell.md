# Phase 3 — PyQt6 Shell + Bridge

## Goal
Desktop application with a chat panel that connects to the Rust agent via stdio
JSON-RPC. User types in the GUI, sees streaming agent responses and tool call
activity. No annotation canvas yet — just the app skeleton and chat.

## What this phase builds

1. **PyQt6 application** — main window with dock-based layout
2. **Process manager** — spawns and monitors Rust agent + Python MCP server
3. **Bridge** — async reader/writer on agent's stdio, JSON-RPC protocol
4. **Chat panel** — markdown rendering, tool call cards, permission dialogs
5. **Agent-side JSON-RPC transport** — replaces the Phase 2 stdin REPL

## Process architecture

```
┌──────────────────────────────────────────────────┐
│  PyQt6 App (Python, main process)                │
│                                                  │
│  ┌──────────────┐   QThread    ┌──────────────┐  │
│  │ MainWindow    │◄───signals──│ AgentBridge   │  │
│  │ ┌──────────┐ │             │ (reader loop) │  │
│  │ │ ChatPanel │ │             └──────┬───────┘  │
│  │ └──────────┘ │                    │           │
│  │ ┌──────────┐ │              stdin/stdout      │
│  │ │ StatusBar │ │                    │           │
│  │ └──────────┘ │                    ▼           │
│  └──────────────┘         ┌──────────────────┐   │
│                           │  tcip-agent       │   │
│                           │  (Rust, sibling    │   │
│                           │   child process)   │   │
│                           └────────┬─────────┘   │
│                                    │ MCP stdio   │
│                                    ▼             │
│                           ┌──────────────────┐   │
│                           │  tcip-mcp-server  │   │
│                           │  (Python, sibling  │   │
│                           │   child process)   │   │
│                           └──────────────────┘   │
└──────────────────────────────────────────────────┘
```

Key: PyQt6 spawns BOTH the Rust agent and the Python MCP server as **sibling child
processes**. It pipes the MCP server's stdio to the agent's MCP input. The GUI
communicates with the agent over a separate JSON-RPC stdio channel.

This sibling model means if the Rust agent crashes, the MCP server (and any running
training) survives. The GUI can restart just the agent and reconnect.

## Package structure

```
tcip-gui/
├── pyproject.toml
├── src/tcip_gui/
│   ├── __init__.py
│   ├── app.py                  # QApplication, MainWindow, dock layout
│   ├── bridge.py               # AgentBridge: QProcess + JSON-RPC read/write
│   ├── protocol.py             # Message types for JSON-RPC (dataclasses)
│   ├── process_manager.py      # Spawn/monitor/restart agent + MCP server
│   ├── panels/
│   │   ├── __init__.py
│   │   ├── chat_panel.py       # Chat history + input + tool cards
│   │   ├── status_bar.py       # Token usage, cost, connection status
│   │   └── placeholder.py      # "Coming soon" placeholder for center panels
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── markdown_view.py    # QTextBrowser or QWebEngineView for markdown
│   │   ├── tool_card.py        # Collapsible tool call display
│   │   ├── permission_card.py  # HITL checkpoint approval dialog
│   │   └── code_block.py       # Syntax-highlighted code display
│   └── resources/
│       ├── style.qss           # Global stylesheet
│       └── icons/              # App icons
└── tests/
    ├── test_bridge.py          # JSON-RPC parsing, message routing
    ├── test_protocol.py        # Message serialization
    └── test_process_manager.py # Spawn/crash/restart behavior
```

## JSON-RPC bridge protocol

Newline-delimited JSON over stdin/stdout. Each message:

```json
{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": "..."}
```

### Agent → GUI messages

| Method | Params | Description |
|--------|--------|-------------|
| `assistant.text_delta` | `{text: str}` | Streaming text chunk |
| `assistant.text_done` | `{text: str}` | Complete text block |
| `tool.call_start` | `{id, name, input}` | Agent invoking a tool |
| `tool.call_result` | `{id, output, is_error}` | Tool execution finished |
| `permission.request` | `{id, tool, input, description, level}` | Needs user approval |
| `canvas.load_image` | `{path, annotations?}` | Show image on canvas |
| `canvas.show_predictions` | `{predictions_path}` | Overlay predictions on canvas |
| `canvas.clear` | `{}` | Clear canvas overlay |
| `canvas.highlight` | `{annotation_ids}` | Highlight specific annotations |
| `status.usage` | `{input_tokens, output_tokens, cost}` | Per-turn usage |
| `status.turn_complete` | `{turn_number}` | Turn finished |
| `error` | `{code, message}` | Agent error |

### GUI → Agent messages

| Method | Params | Description |
|--------|--------|-------------|
| `user.message` | `{text: str}` | User typed a message |
| `permission.response` | `{id, allowed: bool, reason?}` | Approve/deny checkpoint |
| `control.cancel` | `{}` | Cancel current operation |
| `control.shutdown` | `{}` | Graceful shutdown |

### Agent-side changes for JSON-RPC

Replace the Phase 2 `repl.rs` with a `jsonrpc_transport.rs`:
- Reads JSON-RPC from stdin (newline-delimited)
- Writes JSON-RPC to stdout (newline-delimited)
- Routes `user.message` → `ConversationRuntime::run_turn()`
- Emits streaming events during API response
- Emits tool call events during dispatch
- Routes `permission.request` ↔ `permission.response` through `PermissionEnforcer`

The ConversationRuntime doesn't change — only the transport layer swaps.

Note: `canvas.*` messages are generated by the Rust `canvas_control` native tool.
The LLM calls `canvas_control({action: "load_image", ...})`, and the transport
serializes it as a `canvas.*` JSON-RPC message to the GUI.

## Chat panel design

```
┌────────────────────────────────────┐
│  TCIP Agent                   [⚙]  │  ← title bar, settings access
├────────────────────────────────────┤
│                                    │
│  🤖 Agent:                        │
│  "The hazelnut catkin detection    │
│  pipeline would use..."            │
│                                    │
│  ┌─ Tool: get_trait_info ────────┐ │  ← collapsible card
│  │ Input: {crop: "hazelnut"...}  │ │
│  │ Output: {pipeline: ...}       │ │
│  └───────────────────────────────┘ │
│                                    │
│  🤖 Agent:                        │
│  "Based on the registry entry..."  │
│                                    │
│  ┌─ ⚠ Checkpoint: Training ─────┐ │  ← permission card
│  │ Launch training with config:  │ │
│  │ ┌──────────────────────────┐  │ │
│  │ │ model: fasterrcnn_r50    │  │ │
│  │ │ epochs: 100              │  │ │
│  │ │ lr: 0.001                │  │ │
│  │ └──────────────────────────┘  │ │
│  │  [Approve]  [Deny]  [Edit]   │ │
│  └───────────────────────────────┘ │
│                                    │
├────────────────────────────────────┤
│  [Type a message...]         [▶]  │  ← input area
├────────────────────────────────────┤
│  🟢 Agent connected  $0.03  1.2k↑ │  ← status bar
└────────────────────────────────────┘
```

## Application layout (dock-based)

```
┌─────────────────────────────────────────────────────┐
│  [Menu Bar]                                          │
├──────────────┬──────────────────────┬───────────────┤
│              │                      │               │
│  Chat Panel  │  Center Area         │  (future:     │
│  (left dock, │  (placeholder in     │   properties, │
│   ~350px)    │   Phase 3 — shows    │   details)    │
│              │   welcome text)      │               │
│              │                      │               │
│              │                      │               │
│              │                      │               │
│              │                      │               │
├──────────────┴──────────────────────┴───────────────┤
│  Status Bar                                          │
└─────────────────────────────────────────────────────┘
```

The center area stays empty (placeholder) in Phase 3. Phase 4 fills it with
the annotation canvas and review panel.

## Process lifecycle management

```python
class ProcessManager:
    """Spawns and monitors agent + MCP server as sibling child processes."""
    
    def start(self) -> tuple[QProcess, QProcess]:
        # Spawn MCP server: python -m tcip_mcp
        # Spawn agent: tcip-agent --mcp-stdio --jsonrpc
        # Pipe MCP server stdio to agent's MCP input
        # Connect agent's JSON-RPC stdout → bridge.on_data
        # Connect finished signals → crash handlers
        
    def handle_agent_crash(self, exit_code, exit_status):
        # Log crash
        # Show error in chat panel
        # MCP server still alive — training continues
        # Offer restart button (reconnects to same MCP server)
        
    def handle_mcp_crash(self, exit_code, exit_status):
        # Log crash
        # Agent enters degraded mode (native tools only)
        # Offer restart MCP server button
    
    def shutdown(self):
        # Send control.shutdown to agent
        # Send SIGTERM to MCP server
        # Wait for graceful exit (timeout 5s)
        # Kill if still running
```

Agent manages the MCP server lifecycle internally (same as Phase 2).
GUI only manages the agent process.

## What carries over from claw-code

- **Nothing from claw-code's CLI crate** — crossterm rendering is replaced entirely
- **The transport pattern** — claw-code's CLI reads from QueryEngine's async stream.
  We do the same, but over JSON-RPC instead of in-process async iterators.
- **Permission prompting trait** — claw-code's `PermissionPrompter` trait allows
  swapping the approval UI. We implement it for JSON-RPC (agent sends request,
  waits for response from GUI).

## What's new

- All of the GUI code (PyQt6 is a new addition, not from any existing codebase)
- JSON-RPC protocol definition
- Process lifecycle management
- Qt signals connecting bridge events to panel updates

## Test criteria (Phase 3 complete when)

1. `python -m tcip_gui` opens a PyQt6 window with chat panel
2. Agent process starts automatically, MCP server starts via agent
3. Type "Hello" → see agent response streamed in chat
4. Type "What hazelnut traits can be automated?" → agent queries registry via MCP → answer appears
5. Tool calls appear as collapsible cards in chat
6. Permission checkpoint renders as approval card → clicking Approve continues
7. Agent crash → error message in chat → restart button works
8. Status bar shows token count and cost after each turn
9. Closing the window gracefully shuts down agent and MCP server

## Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "PyQt6>=6.5",
    "PyQt6-WebEngine>=6.5",  # for markdown rendering (optional, QTextBrowser fallback)
]
```
