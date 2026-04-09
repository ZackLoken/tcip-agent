# 03 — API Client

## Provider Abstraction

```rust
pub enum ProviderClient {
    Anthropic(AnthropicClient),
    XAI(OpenAICompatClient),
    OpenAI(OpenAICompatClient),
}
```

Dispatch over three providers via `from_model(model_name)` which auto-detects
provider from the model string. Implements both `send_message()` (sync) and
`stream_message()` (SSE streaming).

## SSE Streaming

`SseParser` in `sse.rs`:
- Buffers incoming HTTP chunks byte-by-byte
- Detects frames by double-newline (`\n\n`) separators
- Parses `event:` and `data:` prefixes
- Ignores `ping` events and `[DONE]` markers
- Deserializes JSON to `StreamEvent` enum

### StreamEvent Sequence

```
MessageStart → ContentBlockStart → ContentBlockDelta* → ContentBlockStop →
MessageDelta → MessageStop
```

### OpenAI Normalization

For OpenAI-compatible providers, a normalization layer:
- Buffers tool_call arguments per-tool index
- Converts `ChatCompletion.choice.delta.content` → `ContentBlockDelta.TextDelta`
- Maps `tool_calls` → `ToolUse` content blocks with proper sequencing

## Prompt Cache

`prompt_cache.rs` — Anthropic-specific optimization.

- **FNV-1a fingerprinting** of (model, system, tools, messages) + version
- **Completion cache**: 30s TTL, keyed by request hash
- **Prompt-level tracking**: 5min TTL, detects cache breaks

### Cache Break Detection

- **Expected**: fingerprint version change, parameter change, timeout
- **Unexpected**: cache tokens dropped with stable fingerprint = possible API bug

### Stats

- Per-session cache hits/misses/writes
- Unexpected vs expected invalidation counts
- Total `cache_creation` + `cache_read` tokens
- Last break reason and source

## Error Types

```rust
pub enum ApiError {
    MissingCredentials,
    ContextWindowExceeded,
    ExpiredOAuthToken,
    Api { status, error_type, message, request_id, body, retryable },
    RetriesExhausted { attempts, last_error },
    // + HTTP, IO, JSON, SSE frame errors
}
```

### Failure Classification

`safe_failure_class()` categorizes errors:
- `provider_auth` (401/403, OAuth)
- `provider_rate_limit` (429)
- `provider_internal` (generic fatal)
- `context_window` (token budget)
- `provider_retry_exhausted`
- `provider_transport` (HTTP/SSE)
- `runtime_io` (file/JSON)

### Retry Policy

- Exponential backoff: 200ms initial → 2s max, 2x multiplier
- **Retryable**: HTTP connect/timeout/request, API 5xx
- **Non-retryable**: Auth, context window, JSON, env errors
- Request ID preserved across retries
