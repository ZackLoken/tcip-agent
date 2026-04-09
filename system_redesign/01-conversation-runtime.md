# 01 — Conversation Runtime

The `ConversationRuntime` in `runtime/src/conversation.rs` is the heart of the agent.

## Core Struct

```rust
pub struct ConversationRuntime<C, T>
where
    C: ApiClient,
    T: ToolExecutor,
{
    session: Session,
    api_client: C,
    tool_executor: T,
    permission_policy: PermissionPolicy,
    system_prompt: Vec<String>,
    max_iterations: usize,              // Default: usize::MAX
    usage_tracker: UsageTracker,
    hook_runner: HookRunner,
    auto_compaction_input_tokens_threshold: u32,
    hook_abort_signal: HookAbortSignal,
    hook_progress_reporter: Option<Box<dyn HookProgressReporter>>,
    session_tracer: Option<SessionTracer>,
}
```

Generic over `C: ApiClient` and `T: ToolExecutor` — allows swapping real API for mocks,
and real tool dispatch for in-memory stubs.

## Traits

```rust
pub trait ApiClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError>;
}

pub trait ToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError>;
}
```

Minimal contracts. `ApiClient` returns a vec of events from streaming.
`ToolExecutor` takes a tool name + JSON input string, returns JSON output string.

## AssistantEvent Enum

```rust
pub enum AssistantEvent {
    TextDelta(String),              // Incremental text fragment
    ToolUse { id, name, input },    // Model requests a tool call
    Usage(TokenUsage),              // Token counts for the turn
    PromptCache(PromptCacheEvent),  // Cache hit/miss telemetry
    MessageStop,                    // Stream end marker
}
```

## Main Loop: `run_turn()`

```
1. Push user input → session
2. Retry loop (tracks iterations):
   a. api_client.stream(ApiRequest { system_prompt, messages })
   b. Parse AssistantEvent stream → ConversationMessage (text + tool_uses + usage)
   c. Check for pending ToolUse blocks
   d. If none → break (assistant is done)
   e. For each tool:
      i.   Run PreToolUse hook (can modify input, override permissions, cancel)
      ii.  Run PermissionPolicy::authorize() (may prompt user)
      iii. If allowed: tool_executor.execute(tool_name, input) → capture output/error
      iv.  Run PostToolUse or PostToolUseFailure hook (can inject feedback)
      v.   Push ConversationMessage::tool_result() to session
   f. Check auto-compaction (input_tokens > threshold → compact session)
   g. Record telemetry
3. Return TurnSummary { assistant_messages, tool_results, usage, iterations, auto_compacted }
```

## Key Design Points

### Tool dispatch is synchronous within a turn
Each tool executes sequentially. No concurrent tool execution in the Rust version
(unlike the old TS version which batched read-only tools concurrently). This is simpler
and avoids race conditions.

### Tool errors don't kill the conversation
ToolError is captured as an error result and pushed back to the model. The loop continues.
The model sees the error and can decide what to do (retry, try different approach, ask user).

### Hooks wrap every tool call
Pre-hook fires before permission check. Can:
- Modify tool input
- Override permission decision (Allow/Deny/Ask)
- Cancel the tool call entirely

Post-hook fires after execution. Can:
- Inject feedback into the result
- Log/audit

### Auto-compaction
When input tokens exceed `auto_compaction_input_tokens_threshold`, the session is
compacted (older messages summarized/removed). This prevents context window overflow.

### StaticToolExecutor for testing
```rust
StaticToolExecutor::new()
    .register("add", |input| Ok(parse_and_sum(input)))
```
In-memory executor that maps tool names to closures. Used in unit tests.

## Return Type

```rust
pub struct TurnSummary {
    assistant_messages: Vec<ConversationMessage>,
    tool_results: Vec<ToolResult>,
    usage: TokenUsage,
    iterations: usize,
    auto_compacted: bool,
}
```
