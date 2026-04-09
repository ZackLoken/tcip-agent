# Phase 2 — Rust Agent Core

## Goal
Minimal Rust agent binary that holds multi-turn conversations via Anthropic API,
dispatches native tools (file I/O, bash, grep) and MCP tools (from Phase 1 server),
enforcesHITL permission gates, and supports subagent mode switching.
Testable as a terminal REPL before any GUI exists.

The MCP server is NOT spawned by the agent. In Phase 2 the agent connects to an
already-running MCP server via stdio (launched externally). In Phase 3+ the GUI
spawns both the agent and MCP server as sibling processes.

## What gets adapted from claw-code

### Direct port (modify lightly)
| claw-code module | What it does | Adaptation |
|------------------|-------------|------------|
| `runtime/conversation.rs` | `ConversationRuntime<C,T>`, `run_turn()` loop | Add `checkpoint_requested` to TurnSummary |
| `runtime/config.rs` | 5-level config hierarchy | Reduce to 3 levels: defaults → project → session |
| `runtime/session.rs` | JSONL session persistence, forking | Add TCIP fields (crop, trait, pipeline_stage) to session metadata |
| `api/` crate | AnthropicClient, SSE streaming, prompt caching, retry | Keep as-is — just configure model/keys |
| `tools/bash.rs` | Shell execution with timeout, run-in-background | Keep as-is |
| `tools/file_ops.rs` | read/write/edit with permission checks | Keep as-is |
| `tools/grep.rs`, `tools/glob.rs` | Search tools | Keep as-is |
| `runtime/permission_enforcer.rs` | PermissionEnforcer trait, 5 modes | Simplify to 3 modes: ReadOnly, WorkspaceWrite, FullAccess |

### Substantially rework
| claw-code module | What changes |
|------------------|-------------|
| `runtime/hooks.rs` | Replace shell-script hooks with HITL checkpoint system |
| `cli/` crate | Replace crossterm REPL with simple stdin/stdout for Phase 2 (GUI replaces in Phase 3) |
| `runtime/mcp_subsystem.rs` | Connect to externally-launched MCP server (not spawn it) |
| `plugins/` crate | Drop plugin/worker system entirely — not needed for MVP |

### Drop entirely
- Worker/team/cron tools (Phase 5+ if ever)
- Plugin lifecycle, PluginManager
- Recovery recipes (our scope is simpler)
- LSP integration
- Notebook tools
- Compat harness (build our own test harness)

## Architecture

```
┌─────────────────────────────────────────────┐
│  tcip-agent (Rust binary)                   │
│                                             │
│  ┌─────────────────────┐                    │
│  │ ConversationRuntime  │                   │
│  │  <AnthropicClient,   │                   │
│  │   ToolDispatcher>    │                   │
│  └────────┬────────────┘                    │
│           │                                 │
│  ┌────────▼────────────┐                    │
│  │  ToolDispatcher      │                   │
│  │  ┌──────────────┐   │                    │
│  │  │ NativeTools   │   │  bash, file I/O,  │
│  │  │ (in-process)  │   │  grep, glob       │
│  │  └──────────────┘   │                    │
│  │  ┌──────────────┐   │                    │
│  │  │ McpBridge     │───┼── stdio ──→ Python MCP (external, Phase 1)
│  │  │ (client)      │   │                    │
│  │  └──────────────┘   │                    │
│  └─────────────────────┘                    │
│                                             │
│  ┌─────────────────────┐                    │
│  │ PermissionEnforcer   │                   │
│  │ (HITL checkpoints)  │                    │
│  └─────────────────────┘                    │
│                                             │
│  ┌─────────────────────┐                    │
│  │ SessionManager       │                   │
│  │ (JSONL persistence) │                    │
│  └─────────────────────┘                    │
│                                             │
│  ┌─────────────────────┐                    │
│  │ SkillInjector        │  dynamic system   │
│  │ (system prompt)     │  prompt assembly   │
│  └─────────────────────┘                    │
└─────────────────────────────────────────────┘
```

## Crate structure

```
tcip-agent/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── runtime/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── conversation.rs   # ConversationRuntime<C,T>
│   │   │   ├── config.rs         # 3-level config
│   │   │   ├── session.rs        # JSONL session persistence
│   │   │   ├── permission.rs     # PermissionEnforcer trait + modes
│   │   │   ├── checkpoint.rs     # HITL checkpoint types + resolution
│   │   │   ├── skills.rs         # Dynamic skill injection
│   │   │   └── usage.rs          # Token/cost tracking
│   │   └── Cargo.toml
│   ├── api/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── anthropic.rs      # AnthropicClient (SSE streaming)
│   │   │   ├── types.rs          # Message, ContentBlock, ToolUse, etc.
│   │   │   └── cache.rs          # Prompt caching support
│   │   └── Cargo.toml
│   ├── tools/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── dispatcher.rs     # ToolDispatcher (native + MCP)
│   │   │   ├── bash.rs           # Shell execution
│   │   │   ├── file_ops.rs       # read/write/edit
│   │   │   ├── search.rs         # grep + glob
│   │   │   ├── web.rs            # HTTP fetch
│   │   │   ├── mcp_bridge.rs     # MCP client → external Python server
│   │   │   └── canvas_control.rs # Push actions to GUI canvas (Phase 3+)
│   │   └── Cargo.toml
│   └── cli/
│       ├── src/
│       │   ├── main.rs           # Binary entry point
│       │   └── repl.rs           # Simple stdin/stdout REPL
│       └── Cargo.toml
├── skills/                       # Markdown skill files (injected into system prompt)
│   ├── pipeline-design.md
│   ├── model-selection.md
│   ├── training-config.md
│   ├── image-processing.md
│   ├── annotation-workflow.md
│   ├── crop-science.md
│   └── evaluation-method.md
├── config/
│   └── defaults.toml             # Default config (model, temperature, etc.)
└── tests/
    ├── mock_api.rs               # Mock Anthropic service
    ├── mock_mcp.rs               # Mock MCP server
    ├── test_conversation.rs      # Turn loop tests
    ├── test_tools.rs             # Tool dispatch tests
    ├── test_permissions.rs       # Permission enforcement tests
    └── test_sessions.rs          # JSONL persistence tests
```

## Permission model (standardized vocabulary)

Claw-code has 5 modes. We simplify to 3:

| Mode | Description | Use case |
|------|-------------|----------|
| `ReadOnly` | Registry queries, file reads, grep | Safe browsing |
| `WorkspaceWrite` | + file writes, annotations, inference | Normal work |
| `FullAccess` | + training launch, HPO, model deployment | Needs explicit approval |

HITL checkpoints trigger on **any tool in `FullAccess` category** plus specific
workflow-level gates:
1. **Pipeline design** — show proposed pipeline config, await approval
2. **Pipeline config review** — show training config + augmentation + hyperparameters, await review
3. **Training launch** — show config + cost estimate, await confirmation
4. **Results review** — show metrics + samples, await accept/retrain decision
5. **Model deployment** — show final model + test results, await confirmation

Implementation: `PermissionEnforcer` checks tool permission level. For `FullAccess` tools,
it calls `resolve_checkpoint()` which in Phase 2 blocks on stdin (y/n), and in
Phase 3+ renders a GUI dialog.

## Subagent mode switching

Four subagents, implemented as mode switches (not separate processes). Each mode
swaps system prompt skills + tool filter + model selection.

| Mode | Model | Skills (see [10-skills-architecture](../adaptation/10-skills-architecture.md)) | Tool access |
|------|-------|--------|-------------|
| PipelineDesigner | Opus | pipeline-design, model-selection, crop-science | Registry, data (read-only) |
| CodeGenerator | Sonnet | training-config, image-processing | Registry, data, file I/O |
| TrainingOrchestrator | Sonnet | training-config | Training, inference, model management |
| ResultsAnalyzer | Sonnet | crop-science, evaluation-method | Evaluation, inference, model management, export |

Switching mechanism: The runtime has a `SubagentMode` enum. The agent (or model
via a `switch_mode` tool) activates a mode. The `SkillInjector` reads the active
mode and assembles the appropriate system prompt + tool filter.

The conversation continues in the same session — mode switching only changes what
the model sees in its system prompt and which tools are available.

## System prompt assembly

Follows claw-code's `SystemPromptBuilder` pattern with skill injection added.
See [10-skills-architecture.md](../adaptation/10-skills-architecture.md) for full spec.

```
┌─────────────────────────────────────┐
│  Base system prompt                 │  "You are an ML/CV agent for
│  (always present)                   │   tree crop breeding programs..."
├─────────────────────────────────────┤
│  System rules                       │  Adapted from claw-code's System +
│  (static)                           │  Doing Tasks + Actions sections
├─────────────────────────────────────┤
│  [DYNAMIC_BOUNDARY]                 │
├─────────────────────────────────────┤
│  Active skills (0-3)                │  Injected by SkillInjector based on
│  (from skills/ directory)           │  SubagentMode. 4K chars/file max.
├─────────────────────────────────────┤
│  Project context                    │  From <project>/.tcip/state.toml
│  (crop, traits, pipeline stage)     │  "Working on: hazelnut catkin phenology"
├─────────────────────────────────────┤
│  Registry excerpt                   │  Relevant crops.yml subset for
│  (active crop/traits only)          │  the current pipeline
├─────────────────────────────────────┤
│  Environment                        │  Model name, cwd, date, GPU info
├─────────────────────────────────────┤
│  Tool descriptions                  │  Auto-generated from tool specs
│  (filtered by mode)                 │  (native + MCP, filtered by mode)
└─────────────────────────────────────┘
```

## MCP client integration

In Phase 2, the agent connects to an externally-launched MCP server:

```
User starts MCP server: python -m tcip_mcp (terminal 1)
User starts agent:      tcip-agent --mcp-stdio (terminal 2)
  → Agent sends MCP initialize handshake
  → tools/list → registers MCP tools in tool registry
  → ready for conversation
```

In Phase 3+, the GUI spawns both as sibling child processes and pipes them together.

Degraded mode (from claw-code): if MCP connection breaks, log warning, mark MCP tools
as unavailable, continue with native tools only. Offer to reconnect.

## Configuration (4-level hierarchy)

| Level | Location | Contents |
|-------|----------|----------|
| Defaults | `config/defaults.toml` (compiled in) | model name, temperature, max tokens, permission mode |
| User | `~/.tcip/settings.toml` | API key ref, preferred model, theme |
| Project | `<project_dir>/.tcip/config.toml` | crop, traits, data paths, model preferences |
| CLI flags | Command-line arguments | Runtime overrides (model, permission mode, MCP address) |

## Test criteria (Phase 2 complete when)

1. `tcip-agent` starts, connects to externally-running MCP server, completes tool discovery
2. Terminal REPL conversation: "What traits can be automated for hazelnut?"
   → agent calls `list_crops` / `get_crop_traits` via MCP → coherent answer
3. "What would a catkin phenology pipeline look like?"
   → agent uses pipeline-design skill → structured two-layer pipeline proposal
4. Agent can read/write files via native tools
5. Training tool call → HITL checkpoint fires → user approves/denies via stdin
6. Session persistence: conversation survives agent restart (reload from JSONL)
7. Subagent mode switching works: `switch_mode("TrainingOrchestrator")` changes system prompt + tools
8. Mock API tests pass (no real API calls needed for CI)
9. MCP connection loss → degraded mode → agent continues with native tools

## Key Rust dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.12", features = ["stream"] }
futures = "0.3"
toml = "0.8"
tracing = "0.1"
tracing-subscriber = "0.3"
uuid = { version = "1", features = ["v4"] }
```
