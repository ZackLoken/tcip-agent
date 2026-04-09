# 05 — Sessions & Configuration

## Session Persistence (JSONL)

### Session Struct

```rust
pub struct Session {
    version: u32,                          // VERSION = 1
    session_id: String,
    created_at_ms: u64,
    updated_at_ms: u64,
    messages: Vec<ConversationMessage>,
    compaction: Option<SessionCompaction>,  // Audit trail when context compacted
    fork: Option<SessionFork>,             // Parent session + branch name
    persistence: Option<SessionPersistence>,  // File path
}
```

### Message Types

```rust
pub struct ConversationMessage {
    role: MessageRole,  // System, User, Assistant, Tool
    blocks: Vec<ContentBlock>,
    usage: Option<TokenUsage>,
}

pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: String },
    ToolResult { tool_use_id: String, tool_name: String, output: String, is_error: bool },
}
```

### JSONL Format

```jsonl
{"type":"session_meta","version":1,"session_id":"...","created_at_ms":...}
{"type":"message","message":{...}}
{"type":"message","message":{...}}
{"type":"compaction","count":1,"removed_message_count":42,"summary":"..."}
```

### Operations

- `push_message()` — append atomic JSONL line
- `save_to_path()` — full snapshot (rotation if > 256KB)
- `load_from_path()` — detect JSON vs JSONL, parse linearly
- `fork()` — new session with copied messages, new ID, fork provenance

### Rotation

After pushing, if file > 256KB, rotate:
- `.1`, `.2`, `.3` backup files (keep max 3)
- New writes go to main file

---

## 5-Level Configuration

### ConfigLoader

Files discovered in precedence order (later overrides earlier):

```
1. ~/.claw.json                         (legacy user settings)
2. $CLAW_CONFIG_HOME/settings.json      (user settings)
   or ~/.claw/settings.json
3. .claw.json in cwd                    (project root)
4. .claw/settings.json in cwd           (project nested)
5. .claw/settings.local.json in cwd     (local overrides, highest precedence)
```

**Merge strategy**: Deep merge JSON objects. Later entries override earlier for same keys.

### RuntimeConfig

```rust
pub struct RuntimeConfig {
    merged: BTreeMap<String, JsonValue>,   // Raw merged JSON tree
    loaded_entries: Vec<ConfigEntry>,      // Audit: which files contributed
    feature_config: RuntimeFeatureConfig,  // Parsed typed settings
}
```

### RuntimeFeatureConfig

```rust
pub struct RuntimeFeatureConfig {
    hooks: RuntimeHookConfig,              // pre/post/failure shell commands
    mcp: McpConfigCollection,              // Server name → config map
    oauth: Option<OAuthConfig>,
    model: Option<String>,                 // LLM name override
    permission_mode: Option<ResolvedPermissionMode>,
    permission_rules: RuntimePermissionRuleConfig,  // allow/deny/ask patterns
    sandbox: SandboxConfig,
}
```

### Key Point

Configuration is read-only after boot. The runtime loads, merges, and freezes config
at startup. No hot-reloading.
