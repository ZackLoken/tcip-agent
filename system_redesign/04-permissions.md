# 04 — Permissions

## Permission Modes (Escalating Access)

```rust
pub enum PermissionMode {
    ReadOnly,           // No state changes
    WorkspaceWrite,     // Read + write within workspace bounds
    DangerFullAccess,   // Anything goes
    Prompt,             // Ask user for each tool call
    Allow,              // Auto-allow all
}
```

## PermissionPolicy

```rust
pub struct PermissionPolicy {
    active_mode: PermissionMode,
    tool_requirements: BTreeMap<String, PermissionMode>,  // per-tool minimum
    allow_rules: Vec<PermissionRule>,
    deny_rules: Vec<PermissionRule>,
    ask_rules: Vec<PermissionRule>,
}
```

## Authorization Flow (`authorize_with_context`)

```
1. Deny rules first → if matching pattern → DENY immediately
2. Mode check → active_mode >= required_mode for the tool?
3. Ask rules → matching pattern? → PROMPT user
4. Allow rules → matching pattern? → ALLOW
5. Hook overrides → PermissionOverride::Allow/Deny/Ask can short-circuit
```

## Permission Rules

Glob or substring patterns applied to tool names:
- `"bash"` — exact match
- `"*_write"` — glob pattern

Configured in settings:
```rust
pub struct RuntimePermissionRuleConfig {
    allow: Vec<String>,   // Auto-allow patterns
    deny: Vec<String>,    // Hard-deny patterns
    ask: Vec<String>,     // Require user prompt
}
```

## PermissionPrompter Trait

```rust
pub trait PermissionPrompter {
    fn decide(&mut self, request: &PermissionRequest) -> PermissionPromptDecision;
}

pub enum PermissionPromptDecision {
    Allow,
    Deny { reason: String },
}
```

The CLI implements this with a terminal prompt. A GUI would implement it with a dialog.

## PermissionEnforcer (Higher-Level Wrapper)

```rust
pub fn check(&self, tool_name: &str, input: &str) -> EnforcementResult
pub fn check_file_write(&self, path: &str, workspace_root: &str) -> EnforcementResult
pub fn check_bash(&self, command: &str) -> EnforcementResult
```

Specific checks for file writes (workspace boundaries) and bash commands (read-only
validation, destructive warnings).

## Hook Overrides

Hooks can override permission decisions:

```rust
pub enum PermissionOverride {
    Allow,   // Hook says "safe, skip prompt"
    Deny,    // Hook says "dangerous, block"
    Ask,     // Hook says "ask the user"
}
```

Passed via `PermissionContext` during the conversation loop. Pre-hook fires before
the permission check, so it can preempt the normal authorization flow.

## Lifecycle Hooks

```rust
pub struct HookRunner {
    config: RuntimeHookConfig,  // pre_tool_use, post_tool_use, post_tool_use_failure
}
```

Hooks are shell commands that receive JSON (tool_name, input) on stdin and return
JSON (permission_override, updated_input, messages) on stdout.

```rust
pub struct HookRunResult {
    denied: bool,
    failed: bool,
    cancelled: bool,
    messages: Vec<String>,
    permission_override: Option<PermissionOverride>,
    updated_input: Option<String>,
}
```

**Pre-hook** can:
- Modify tool input before permission check
- Override permission decision
- Cancel the tool call

**Post-hook** can:
- Inject feedback into the result
- Log/audit the execution
