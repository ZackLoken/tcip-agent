# Build Plan — Index

5-phase build sequence. Each phase is independently testable.

## Structure

Monorepo with three Python packages + Rust agent:
```
tcip/
├── packages/
│   ├── tcip-annotation/    # Shared annotation engine (MCP + GUI)
│   ├── tcip-mcp/           # Python MCP server (domain tools)
│   └── tcip-gui/           # PyQt6 desktop application
├── tcip-agent/             # Rust agent core (adapted from claw-code)
├── registry/               # crops.yml (source of truth)
└── skills/                 # Markdown skill files for subagents
```

## Prerequisites

- Hazelnut catkin images: AVAILABLE (user provides path)
- Hazelnut catkin annotations (YOLO format): AVAILABLE
- Trained hazelnut catkin detection model: AVAILABLE
- claw-code source: available locally (see [00-reference-index.md](00-reference-index.md))

### Files that must exist in the new repo alongside these docs

| File/Dir | Source | Purpose |
|----------|--------|---------|
| `registry/crops.yml` | Copied from design repo | 180 traits × 6 crops, YAML source of truth |
| `skills/*.md` | To be consolidated (see [10-skills-architecture](adaptation/10-skills-architecture.md)) | 7 skill files for agent system prompt injection |
| `docs/system_redesign/` | This directory | Complete implementation specification |

## Phases

| # | Phase | Deliverable | Test Criteria |
|---|-------|-------------|---------------|
| 1 | [Python MCP Server](phases/01-mcp-server.md) | Domain tools accessible via MCP protocol | MCP CLI client can query registry, load dataset, run inference |
| 2 | [Rust Agent Core](phases/02-rust-agent.md) | Agent holds conversations, calls MCP tools | Terminal REPL: ask about hazelnut traits, agent queries registry |
| 3 | [PyQt6 Shell + Bridge](phases/03-gui-shell.md) | Desktop app with chat panel connected to agent | Type in GUI chat, see agent responses + tool calls |
| 4 | [Annotation Canvas](phases/04-annotation-canvas.md) | Full annotation + review in GUI | Load hazelnut images, annotate, review predictions |
| 5 | [Full Loop](phases/05-full-loop.md) | End-to-end pipeline on hazelnut catkin phenology | Two-layer pipeline (bush isolation → catkin detection+classification → temporal aggregation) → 4 phenology traits → per-plant CSV |

## Dependency Graph

```
Phase 1 (MCP Server)
    ↓
Phase 2 (Rust Agent) ← connects to Phase 1
    ↓
Phase 3 (GUI Shell) ← embeds Phase 2
    ↓
Phase 4 (Annotation) ← renders in Phase 3, calls Phase 1 tools
    ↓
Phase 5 (Full Loop) ← integrates all phases
```
