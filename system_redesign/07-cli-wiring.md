# 07 — CLI Wiring & REPL

## Entry Point

`rusty-claude-cli/src/main.rs` — ~500 LOC.

### CLI Actions

| Action | Trigger | Handler |
|--------|---------|---------|
| REPL | No args | `run_repl()` — interactive loop |
| Prompt | `-p "text"` or `claw "text"` | `run_turn_with_output()` — single turn |
| Resume | `--resume SESSION.jsonl` | Load session + optional new prompt |
| Status | `claw status` | JSON/text snapshot of model, perms, sandbox |
| Doctor | `claw doctor` | Preflight diagnostics |
| Login/Logout | `/login`, `/logout` | OAuth flow |

### CLI Options

- `--model` — claude-opus-4-6 (default; aliases: opus, sonnet, haiku)
- `--output-format text|json`
- `--permission-mode read-only|workspace-write|danger-full-access`
- `--allowedTools bash,read_file,grep_search` — tool whitelist
- `--dangerously-skip-permissions` — disable enforcer

## LiveCli Struct

```rust
struct LiveCli {
    model: String,
    client: ProviderClient,
    runtime: ConversationRuntime,
    renderer: TerminalRenderer,
    session: Session,
    tool_registry: GlobalToolRegistry,
    permitted_tool_set: Option<BTreeSet<String>>,
    permission_mode: PermissionMode,
}
```

This is where everything gets wired together. Config → client → runtime → renderer.

## REPL Loop

```
1. Initialize LiveCli with config
2. Load session (or create fresh)
3. Loop:
   a. Read user input (rustyline with slash-command completions)
   b. Add as InputMessage to session
   c. Build MessageRequest (model, messages, tools, system)
   d. Stream message via ProviderClient
   e. Render streaming events (text deltas → markdown → ANSI)
   f. FOR EACH ToolUse event:
      - Check permission enforcer
      - Execute tool via GlobalToolRegistry
      - Add ToolResult to session
      - Loop back (multi-turn agentic)
   g. ON MessageStop:
      - Save session to JSONL
      - Await next input
```

## Rendering

`render.rs` — Markdown stream renderer.

### MarkdownStreamState

Accumulates `ContentBlockDelta` events and renders markdown → ANSI on-the-fly.
Uses `syntect` for syntax highlighting (base16-ocean.dark theme).

### Color Theme

| Element | Color |
|---------|-------|
| Heading | cyan |
| Strong  | yellow |
| Emphasis | magenta |
| Inline code | green |
| Links | blue |
| Quotes/borders | grey |

### Output Modes

- **Text**: Full ANSI markdown rendered to stdout
- **JSON**: Structured metadata (for programmatic consumption)

## Line Input

`input.rs` — `LineEditor` wrapping `rustyline`:
- Slash command completion
- Emacs edit mode (Ctrl-J for newline)
- History persistence
- Returns `ReadOutcome::Submit(input)`, `Cancel`, or `Exit`
