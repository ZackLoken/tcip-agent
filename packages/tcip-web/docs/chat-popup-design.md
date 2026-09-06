# In-app agent chat pop-up: design

Status: SUPERSEDED (2026-07-09), replaced by the embedded agent terminal
(decided with Zack after the translation-layer chat shipped and under-delivered).
The in-app agent surface is now the *real* Claude Code CLI running in a server-side
PTY (`tcip_web/terminal.py` + `routes/terminal.py`, ConPTY via `pywinpty` on Windows,
stdlib `pty` on POSIX), streamed raw to an xterm.js terminal in a docked rail
(`frontend/src/components/TerminalRail.tsx`) themed to the app palette via the ANSI
slots. Rationale: the audience is tree-crop breeders using Claude as their ML/CV
engineer; they need the full Claude Code experience (skills, CLAUDE.md, permission
prompts, slash commands), and a translated chat envelope is a lossy re-implementation
whose failure mode is silence. What survives from this design: the trust boundary
(loopback + Origin-checked WS), one-session-attach semantics, replay-then-tail
delivery, kill-on-shutdown, the two-MCP-instance concurrency invariant
(`tests/test_concurrent_audit_appends.py`), and the agent→GUI channel (panel events +
`activate_project`), which is independent of how the conversation renders.
End-to-end check: `tools/smoke_terminal_e2e.py` (drives the real CLI).

The remainder of this document is the original chat-popup design, kept for the
option analysis (§2) and the invariants that carried over.

## 1. Problem

The human works in the tcip-web GUI (Annotate / Review / Training / …) while the
Claude agent lives in a separate surface (Claude Code, or any MCP client) attached to
`tcip-mcp` over stdio. To ask the agent something (for example, "why did mAP drop
on the 06-20 date?" or "prioritize a review queue for me") the human must alt-tab to
the terminal.
The deferred item is an in-app chat pop-up: a floating conversation panel inside the
GUI that talks to a Claude agent which can drive the TCIP MCP tools.

The hard part is not the chat UI. It is that nothing in the current system runs an
agent loop:

- `tcip-mcp` is a transport-neutral stdio MCP server. It exposes tools; it does
  not call a model. It is spawned by whatever MCP client connects (`.mcp.json` uses
  `conda run … python -m tcip_mcp`).
- `tcip-web` is a FastAPI backend + React frontend. Its only agent-facing surface is
  one-way, agent → browser: MCP tools POST panel events to
  `/api/events/{panel}` (via `tcip_mcp.web_client.post_panel_event`), the backend
  fans them out over `/ws/panel/{panel}`, and browsers render them
  (`pushAgentActivity` in the store). There is no browser → agent channel.
- A browser cannot spawn a stdio subprocess, so the agent loop has to live in (or
  next to) the tcip-web backend process.

So the feature decomposes into: (a) something that runs the Claude loop and holds
an MCP client connection to `tcip-mcp`, (b) an HTTP/WS chat surface on tcip-web,
and (c) the pop-up UI. This doc covers all three; only (c) has a stub today.

## 2. Attachment options for an agent-driving backend

The MCP server being stdio-only is not actually the constraint people expect: stdio
is the *easy* transport for a locally-spawned subprocess, which is exactly what every
option below does. The real design choice is who runs the model loop.

### Option A: Claude Agent SDK sidecar, managed by tcip-web (recommended)

tcip-web gains an `agent_host` module that, per chat session, drives the Claude Agent
SDK (the Python `claude-agent-sdk` package, which wraps the locally installed Claude
Code runtime in headless mode). The SDK process is pointed at the repo root so it
picks up `.mcp.json` (spawning its own `tcip-mcp` stdio subprocess), `CLAUDE.md`, and the
generated skills under `.claude/skills/` exactly like an interactive Claude Code session.

- Pros
  - Reuses the operator's existing Claude Code auth (subscription or `ANTHROPIC_API_KEY`);
    tcip-web never holds a key of its own.
  - The chat agent *is* the same agent the project is designed around: CLAUDE.md
    operating contract, skills, `@audited` MCP tools, `.tcip/` state, friction
    reports. Zero duplicated agent behavior.
  - Permission prompts map cleanly: the SDK surfaces tool-permission requests as
    structured callbacks → forward over the chat WS → human approves in the pop-up.
    This preserves the "confirm before destructive actions" invariant with the human
    in the GUI, not the terminal.
  - Streaming output (assistant deltas, tool-use events) comes for free as
    structured JSON messages.
- Cons
  - Runtime dependency on an installed/authenticated Claude Code (`claude` CLI).
    Must be detected at startup and reported cleanly when absent (chat button shows
    "agent backend not configured", never a broken panel).
  - Process lifecycle on Windows (orphan cleanup, kill-on-shutdown) needs care,
    same class of problem the training-job runner already solves in `jobstore`.
  - A second `tcip-mcp` instance runs while a chat session is live. This is safe by
    design: all durable state goes through `.tcip/` (append-only `audit.jsonl`,
    immutable experiment dirs), but it is worth a line in the docs and a test.

### Option B: direct Anthropic API loop inside tcip-web

tcip-web embeds its own agent loop: `anthropic` SDK for the model + the `mcp` Python
client SDK to spawn `tcip-mcp` over stdio and translate MCP tools into API tool
definitions.

- Pros: no dependency on Claude Code being installed; fully controllable loop
  (custom system prompt, custom permission gating); easiest to unit test.
- Cons: tcip-web must hold an `ANTHROPIC_API_KEY` (new secret surface in a
  GUI-serving process); re-implements what Claude Code already provides (CLAUDE.md
  ingestion, skills, session resume, context compaction); the chat agent would
  diverge behaviorally from the terminal agent: two agents with different contracts
  operating on the same scientific state is exactly the kind of dual code path this
  repo forbids.

### Option C: relay to the already-running terminal agent

No new agent runtime: tcip-web stores chat messages; the human's existing Claude
Code session polls them via an MCP tool (`get_chat_messages`) and replies via
another (`post_chat_reply`).

- Pros: zero new auth, zero new processes; trivially cheap.
- Cons: MCP stdio has no server → client push, so the terminal agent only sees
  messages when *it* chooses to poll: a chat where the other side answers "whenever
  it next runs a tool" is not a chat. It also couples the GUI feature to a terminal
  session being alive. Rejected as the primary design; the message-store half of it
  is however identical to Phase C1 below, so nothing is wasted if we ever want it as
  a degraded mode.

### Option D: run `tcip-mcp` as streamable-HTTP and add a separate agent-host service

MCPServer can serve streamable HTTP instead of stdio, so a long-lived `tcip-mcp` could
be shared by the terminal client and a new agent-host daemon.

- Pros: single MCP server instance; network-reachable MCP if that's ever wanted.
- Cons: changes the transport story for *every* client ( `.mcp.json`, docs,
  startup ordering) to solve a problem Option A solves without touching it: the
  transport was never the blocker, the missing model loop was. Also adds a third
  long-running process. Rejected for now; revisit only if MCP-over-HTTP becomes a
  requirement independently.

Decision: Option A. One agent identity, no new secrets, permission flow intact.
Option B is the documented fallback if the Claude Code dependency ever becomes
unacceptable.

## 3. HTTP/WS surface (tcip-web)

New router `tcip_web/routes/chat.py`, mounted like the other tab routes. Everything
lives under the existing trust boundary (loopback bind, Origin check on WS,
TrustedHost middleware). Token auth for a network-exposed GUI stays out of scope.

```
POST   /api/chat/sessions                  → {session_id}         create session (spawns sidecar lazily)
GET    /api/chat/sessions                  → [{id, title, state}] list (rehydrated transcripts read-only)
GET    /api/chat/sessions/{id}/messages    → [Message]            transcript replay for reconnect
POST   /api/chat/sessions/{id}/messages    → 202                  user message → sidecar stdin
POST   /api/chat/sessions/{id}/interrupt   → 202                  cancel current agent turn
POST   /api/chat/sessions/{id}/permission  → 200                  {request_id, decision: "allow"|"deny", note?}
GET    /api/chat/status                    → {available, reason?} sidecar preflight (CLI found + authed)
WS     /ws/chat/{id}                       → server-push stream   (Origin-checked like /ws/state)
```

WS message envelope mirrors the sidecar's structured events, minimally:

```jsonc
{"type": "assistant_delta",  "text": "..."}                       // streaming text
{"type": "tool_use",         "name": "launch_training", "input_summary": "..."}
{"type": "tool_result",      "name": "...", "ok": true}
{"type": "permission_request","request_id": "...", "tool": "...", "detail": "..."}
{"type": "turn_done",        "stop_reason": "end_turn"}
{"type": "session_state",    "state": "idle" | "running" | "dead", "reason?": "..."}
```

Design rules, matching the existing panel-event hub:

- Replay on reconnect: transcript is the source of truth (`GET …/messages`), the
  WS is a live tail, same pattern as `/ws/panel/{panel}`'s recent-events deque, but
  persisted to `.tcip/chat/<session_id>.jsonl` so a backend restart yields a
  readable (not resumable) history, exactly like interrupted training jobs in
  `jobstore`.
- One sidecar per session, sessions are cheap to keep few: default UI offers a
  single ongoing session; the API is session-plural so this doesn't need a redesign
  later.
- Permission requests block the sidecar, not the backend: unanswered requests
  time out to "deny" after a generous interval, and the denial reason is fed back to
  the agent (so it can report friction instead of hanging).
- Auditability: the agent's *state changes* are already audited: they go through
  `@audited` MCP tools inside `tcip-mcp`. The chat transcript itself is operator
  I/O, stored under `.tcip/chat/` but not written to `audit.jsonl`.

## 4. UI / UX

A pop-up, not a tab: the whole point is asking questions *without leaving* the
current tab.

- Launcher: small floating button, bottom-right, above the StatusBar
  (`fixed`, z-order under modals, over canvas). Badge dot when the agent produced
  output while the panel was closed.
- Panel: ~380px wide card anchored to the button; message list, input box,
  Send / Stop. Esc closes the panel only (never cancels the agent turn; Stop does).
- Message rendering: user text, streamed assistant text, and *compact* tool-call
  chips ("▸ launch_training: ok") that expand on click. Tool chips keep the
  transcript scannable; the full JSON stays in the transcript file.
- Permission prompts render inline as blocking cards with Allow / Deny +
  optional note; the chat cannot proceed past one silently, mirroring the
  "approval for one doesn't extend to the next" invariant.
- Context chip: the composer shows what the backend will attach to the message
  (active tab, dataset root/date, current image on Annotate/Review). One click
  removes it. This reuses `GuiState`; the backend already holds it in
  `state.store`, so context attachment is a backend concern, not a frontend
  serializer.
- Unconfigured state: if `GET /api/chat/status` says unavailable, the panel
  shows a one-paragraph explanation (install/auth Claude Code) instead of a
  composer. The launcher is still visible so the feature is discoverable.
- Theme/layout: existing Tailwind tokens (`tcip-panel`, `tcip-border`,
  `tcip-accent`, …); no new design system.

## 5. Phased implementation plan

Each phase is independently shippable and reviewable; later phases don't rewrite
earlier ones.

- C0 (this change): design doc + disabled UI shell.
  `ChatPopup` behind `CHAT_POPUP_ENABLED = false`; renders `null`; unit-tested both
  ways. No backend code.
- C1: chat surface, no agent. `routes/chat.py` with the session store,
  transcript persistence in `.tcip/chat/`, WS tail with replay, and a
  `status = unavailable` preflight. Frontend: flip the flag, wire the panel to the
  API, render the unconfigured state. End state: real plumbing, honest "no agent
  yet" UX. (No echo bot: a fake agent is a dual code path we'd have to rip out.)
- C2: Agent SDK sidecar. `agent_host.py`: preflight detection, spawn per
  session (cwd = project root so `.mcp.json` / CLAUDE.md / skills apply), stream
  events → WS envelope, interrupt, kill-on-shutdown (reuse `jobstore` lifecycle
  patterns). Permission callbacks → `permission_request` WS event → decision
  endpoint. Tests with a scripted fake sidecar process (test double at the process
  boundary, not a code path in prod).
- C3: context attachment + deep links. Backend prepends the GuiState-derived
  context block to user messages (with the UI chip from §4). Agent-side: a small
  MCP tool or message convention so the agent can tell the GUI "look here"
  (navigate tab / select image), delivered through the existing panel-event hub
  rather than a new channel.
- C4: polish. Session list/resume UI, transcript export, badge/notification
  behavior, keyboard shortcut, and a `visual-refresh`-pass on the panel once the
  separate visual refresh item lands.

## 6. Invariants and risks

- Science invariants are inherited, not re-implemented: every state mutation the
  chat agent performs goes through the same `@audited` MCP tools; experiments stay
  immutable; destructive actions require the in-chat permission card. The chat adds
  a *transport* for the human↔agent conversation, not a new mutation path.
- Two `tcip-mcp` instances (terminal + sidecar) writing `.tcip/` concurrently:
  audit log is append-only JSONL (atomic-enough line appends), experiments are new
  directories per run. Phase C2 must add a test for concurrent `report_friction` /
  `audit` appends before shipping.
- Feature flag discipline: exactly one flag (`CHAT_POPUP_ENABLED`), removed
  (not inverted, not layered) when C2 ships. No parallel legacy path at any phase.
- Out of scope, unchanged: N-channel/NPZ/GeoTIFF display; SAM auto-label→accept
  GUI flow; token auth for a network-exposed GUI (chat inherits the loopback trust
  boundary and must be re-reviewed if that ever changes).
