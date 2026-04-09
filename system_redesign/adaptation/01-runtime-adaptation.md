# 01 — Runtime Adaptation

## What claw-code has

`ConversationRuntime<C: ApiClient, T: ToolExecutor>` — generic agent loop.
- Push user input → stream API → dispatch tools → loop until done
- Hooks wrap every tool call (pre/post)
- Auto-compaction when nearing token limits
- Sequential tool dispatch (no concurrency)
- Returns `TurnSummary` with messages, results, usage

## What carries over as-is

- **The generic trait pattern.** `ApiClient` and `ToolExecutor` as traits is the
  right abstraction. We keep this exactly. Swap implementations for testing.
- **The main loop structure.** User input → API stream → tool dispatch → loop.
  This is the universal agentic pattern. No reason to change it.
- **Auto-compaction.** Context window management is essential regardless of domain.
- **Tool errors don't kill the conversation.** Model sees the error, decides next step.
- **UsageTracker / telemetry.** Cost tracking matters — API calls add up.

## What needs modification

### Hook system → HITL checkpoints
Claw-code hooks are shell commands for pre/post tool execution. We need something
richer: **Human-in-the-Loop checkpoints** at specific pipeline stages.

Claw-code hooks: "before any bash command, run this shell script"
TCIP needs: "before launching training, pause and show the user the config for approval"

The hook mechanism is the right *place* in the architecture (it wraps tool execution),
but the *implementation* needs to support:
- GUI-rendered approval dialogs (not shell scripts)
- Structured checkpoint context (config YAML, metrics, sample predictions)
- Checkpoint history/audit trail
- Async resolution (user might not respond immediately)

### System prompt → dynamic skill injection
Claw-code uses a static `system_prompt: Vec<String>`. We need dynamic injection
of relevant ML domain knowledge before each turn, based on:
- Which subagent is active (PipelineDesigner vs TrainingOrchestrator)
- What phase of the workflow we're in
- What crop/trait is being worked on

The `Vec<String>` type is fine — we just populate it differently.

### TurnSummary → richer return type
Claw-code's `TurnSummary` has messages, results, usage, iterations.
We may need additional fields:
- `checkpoint_requested: Option<CheckpointType>` — if the agent wants human approval
- `canvas_updates: Vec<CanvasAction>` — if the agent wants to push predictions to GUI
- `training_status: Option<TrainingProgress>` — if training is running

**Alternative**: keep TurnSummary lean, put these in tool results. The GUI inspects
tool results and reacts. This is simpler and more aligned with claw-code's design.
Leaning toward this approach.

## What's new (not in claw-code)

### Long-running operations
Claw-code tools are fast (file reads, grep, bash). ML training runs for hours.
The runtime needs to handle:
- Tool returns immediately with a `run_id`
- Agent polls status via another tool
- GUI shows training dashboard in parallel

This is actually already supported by claw-code's pattern: bash has `run_in_background`
which returns a task ID. We follow the same pattern for `launch_training`.

### Multi-turn workflows with state
A pipeline design → code review → training → evaluation workflow spans many
conversation turns. The session persistence (JSONL) handles this naturally.
But we may want workflow-level state tracking beyond the conversation:
- "We're in the hazelnut catkin project, at the training stage"
- "The last training run was run_id=xyz with mAP=0.72"

**Resolution**: Two tiers.
- **Project state** (`<project>/.tcip/state.toml`): Durable pipeline state that persists
  across sessions — crop, trait set, data paths, current pipeline stage, registered model
  versions. The agent reads this at session start (injected into the system prompt's
  project context block) and updates it through MCP tools (`update_project_state`).
- **Session state** (JSONL session messages): Conversational state — what was discussed,
  decisions made, run_ids from this session, intermediate results.
