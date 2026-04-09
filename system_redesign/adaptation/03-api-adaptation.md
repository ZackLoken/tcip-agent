# 03 — API Client Adaptation

## What claw-code has

ProviderClient with Anthropic + OpenAI-compat backends. SSE streaming.
Prompt cache (Anthropic-only). Retry with exponential backoff. Error classification.

## What carries over as-is

- **AnthropicClient** — we're using direct Anthropic API. Keep the streaming SSE
  client exactly as claw-code has it.
- **Retry policy** — exponential backoff (200ms → 2s). Same logic applies.
- **Error classification** — `safe_failure_class()` categories are universal.
- **Prompt cache** — FNV fingerprinting + TTL tracking. Free performance win.
- **TokenUsage tracking** — essential for cost control.

## What we drop

- **OpenAI-compat provider** — not needed for MVP. Can add later.
- **xAI/Grok provider** — not needed.

## What needs modification

### Model selection per subagent
Claw-code uses one model per session (set via `--model`). We want different models
for different subagents:
- PipelineDesigner → Opus (deep reasoning for architecture decisions)
- CodeGenerator → Sonnet (routine code generation)
- TrainingOrchestrator → Sonnet
- ResultsAnalyser → Sonnet

The `ApiClient` trait doesn't care — it receives the model in `ApiRequest`. The
subagent spawner just passes a different model string. Simple change.

### Subagent specification (mode switches, not separate processes)

| Subagent | Model | Skills | Tool Access |
|----------|-------|--------|-------------|
| PipelineDesigner | Opus | pipeline-design, model-selection, crop-science | Registry queries, data tools (ReadOnly) |
| CodeGenerator | Sonnet | training-config, image-processing | File I/O, registry, data tools (WorkspaceWrite) |
| TrainingOrchestrator | Sonnet | training-config | Training, inference, model tools (FullAccess) |
| ResultsAnalyzer | Sonnet | crop-science, evaluation-method | Evaluation, model, inference tools (WorkspaceWrite) |

Switching mechanism: the runtime's `SkillInjector` swaps the system prompt's skill
block + tool filter when the agent activates a different subagent mode. The main
agent decides which mode to activate based on conversation phase. No separate processes,
no Task/Worker system — just a different system prompt + tool subset + model.

### Context window budget awareness
For ML conversations that span many tool calls (annotation review, training
iterations), we need to be more aggressive about compaction. The auto-compaction
threshold should be configurable per-project or per-task complexity.

## Open question

**Streaming to GUI**: Claw-code streams text deltas to a terminal renderer. We need
to stream them to a PyQt6 chat panel instead. The `AssistantEvent::TextDelta` events
need to cross the stdio JSON-RPC bridge to the GUI process. This is a rendering
concern, not an API client concern — the API client stays the same.
