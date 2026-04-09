# 07 — CLI → GUI Adaptation

## What claw-code has

Terminal REPL: rustyline input → QueryEngine → MarkdownStreamState renderer.
`LiveCli` struct wires config + client + runtime + renderer + session + tools.

## What replaces it

PyQt6 desktop application that embeds the agent via stdio JSON-RPC.

### Process architecture

```
PyQt6 App (Python, main process)
├── Spawns: Rust agent (child process, stdio JSON-RPC)
├── Spawns: Python MCP server (child process, stdio MCP)
├── GUI thread: renders panels, handles user interaction
└── Bridge thread: reads agent stdout, dispatches to GUI signals
```

### Bridge protocol (stdio JSON-RPC, newline-delimited JSON)

Agent → GUI:
- `assistant_text` — streaming text delta
- `assistant_text_done` — text block complete
- `tool_use` — agent is calling a tool (name, input, id)
- `tool_result` — tool execution finished (id, output, is_error)
- `permission_request` — agent needs approval (id, tool, input, description)
- `status` — token usage, cost, turn count
- `error` — agent-side error

GUI → Agent:
- `user_message` — user typed something in chat
- `permission_response` — allow/deny for a permission request
- `cancel` — abort current operation

### GUI panels

**Chat panel** (left dock, always visible):
- Renders agent text as markdown
- Shows tool calls inline (collapsible)
- Permission/checkpoint requests as interactive cards
- User text input at bottom

**Annotation canvas** (center, mode-dependent):
- Built on shared `tcip_annotation.AnnotationEngine` (headless, from Phase 1)
- Box + polygon drawing, vertex editing, streaming, snapping
- Agent can push predictions onto canvas via tool results
- Undo/redo, pan/zoom, EXIF auto-correct

**Review panel** (center, mode-dependent):
- Built on shared `tcip_annotation.ReviewEngine` (headless, from Phase 1)
- IoU-based matching, TP/FP/FN cycling
- Accept/Edit/Reject actions
- Agent triggers review after inference

**Training dashboard** (center, mode-dependent):
- Loss/metric curves (reads TensorBoard logs or agent status updates)
- Stage progress (which unfreezing stage, epoch N/M)
- HPO trial overview if running

**Dataset browser** (secondary panel):
- Image thumbnails with annotation status
- Class distribution chart
- Filter by completion status

### What's NOT in claw-code that the GUI needs

- Canvas rendering (2D annotation). Entirely new.
- Training visualization. New.
- Dataset browsing. New.
- Multiple center-panel modes. New.

The chat panel is the closest analog to claw-code's terminal REPL, just rendered
in PyQt6 instead of crossterm.
