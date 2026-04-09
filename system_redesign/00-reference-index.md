# Claw-Code Architecture Reference

> Source: claw-code (Rust rewrite of Claude Code CLI — available locally as reference)
> Version: Rust rewrite (2026-04-03, 292 commits, 48.6K LOC)

## Overview

Claw-code is Anthropic's Claude Code CLI.
It is a terminal-based agentic coding assistant. This document captures the full
architecture for use as a reference when adapting it into the TCIP phenotyping platform.

## Repository Structure

```
claw-code-main/
├── rust/                         # Canonical Rust workspace (9 crates)
│   ├── crates/
│   │   ├── rusty-claude-cli/     # CLI binary entry point
│   │   ├── runtime/              # Core agent loop, tools, sessions, config, MCP, permissions
│   │   ├── api/                  # HTTP streaming client (Anthropic + OpenAI-compat)
│   │   ├── commands/             # Slash commands
│   │   ├── tools/                # Tool registry + 40 tool specs
│   │   ├── plugins/              # Plugin system
│   │   ├── telemetry/            # Cost tracking
│   │   ├── mock-anthropic-service/  # Deterministic testing
│   │   └── compat-harness/       # Parity validation
│   └── Cargo.toml                # Workspace root
├── src/                          # Python reference/audit code (NOT runtime)
├── PHILOSOPHY.md                 # Architectural vision
├── PARITY.md                     # Rust port status
└── ROADMAP.md                    # Upcoming features
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Rust 2021 edition |
| Async runtime | tokio (multi-threaded) |
| Terminal rendering | crossterm + pulldown-cmark + syntect |
| HTTP client | reqwest (SSE streaming) |
| Serialization | serde + serde_json |
| Line editing | rustyline |
| Linting | forbid(unsafe_code), warn(clippy::all) |

## Document Index

1. [01-conversation-runtime.md](01-conversation-runtime.md) — The agent loop
2. [02-tool-system.md](02-tool-system.md) — Tool registry, native tools, MCP bridge
3. [03-api-client.md](03-api-client.md) — Streaming, providers, prompt cache
4. [04-permissions.md](04-permissions.md) — Permission modes, enforcement, hooks
5. [05-sessions-config.md](05-sessions-config.md) — JSONL persistence, 5-level config
6. [06-mcp-subsystem.md](06-mcp-subsystem.md) — MCP server lifecycle, degraded mode
7. [07-cli-wiring.md](07-cli-wiring.md) — Entry point, REPL loop, rendering
8. [08-recovery-plugins.md](08-recovery-plugins.md) — Failure handling, plugin lifecycle
