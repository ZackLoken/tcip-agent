# 08 — Recovery, Plugins, Workers

## Recovery Recipes

`recovery_recipes.rs` defines structured failure handling.

### Failure Scenarios

```rust
pub enum FailureScenario {
    TrustPromptUnresolved,     // User needs to unlock worker
    PromptMisdelivery,         // Prompt sent to wrong target
    StaleBranch,               // Branch needs rebase
    CompileRedCrossCrate,      // Build failure
    McpHandshakeFailure,       // MCP protocol error
    PartialPluginStartup,      // Some MCP servers failed
    ProviderFailure,           // Model API error
}
```

### Recovery Steps

```rust
pub enum RecoveryStep {
    AcceptTrustPrompt,
    RedirectPromptToAgent,
    RebaseBranch,
    CleanBuild,
    RetryMcpHandshake { timeout },
    RestartPlugin { name },
    RestartWorker,
    EscalateToHuman { reason },
}
```

### Recipe Structure

```rust
pub struct RecoveryRecipe {
    scenario: FailureScenario,
    steps: Vec<RecoveryStep>,
    max_attempts: u32,
    escalation_policy: EscalationPolicy,  // AlertHuman, LogAndContinue, Abort
}
```

Each scenario maps to a hardcoded recipe via `recipe_for(scenario)`.

---

## Plugin Lifecycle

`plugin_lifecycle.rs` — Health and state tracking for external integrations.

### States

```rust
pub enum PluginState {
    Unconfigured,
    Validated,
    Starting,
    Healthy,
    Degraded { healthy_servers, failed_servers },
    Failed { reason },
    ShuttingDown,
    Stopped,
}
```

Key: **Degraded** is a first-class state. The system continues operating with
partial plugin availability rather than failing entirely.

---

## Worker System

`worker_boot.rs` — Sub-agent process lifecycle.

### Worker Status

```rust
pub enum WorkerStatus {
    Spawning,
    TrustRequired,
    ReadyForPrompt,
    Running,
    Finished,
    Failed,
}
```

### Worker Struct

```rust
pub struct Worker {
    worker_id: String,
    cwd: String,
    status: WorkerStatus,
    trust_auto_resolve: bool,
    trust_gate_cleared: bool,
    auto_recover_prompt_misdelivery: bool,
    events: Vec<WorkerEvent>,
    last_error: Option<WorkerFailure>,
}
```

### Event Sequence

```
Spawning → TrustRequired → TrustResolved → ReadyForPrompt → Running → Finished
                                                                    → Failed
```

Workers require trust resolution before accepting prompts. This is the
permission gate for sub-agents — the parent agent (or user) must approve.

---

## Task Registry

`task_registry.rs` — In-memory lifecycle for sub-agent tasks.

```rust
pub struct Task {
    task_id: String,
    prompt: String,
    status: TaskStatus,  // Created, Running, Completed, Failed, Stopped
    messages: Vec<TaskMessage>,
    output: String,
    team_id: Option<String>,
}
```

Thread-safe via `Arc<Mutex<>>`:

```rust
pub fn create(&self, prompt, description) -> Task
pub fn get(&self, task_id) -> Option<Task>
pub fn list(&self, status_filter) -> Vec<Task>
pub fn update(&self, task_id, message) -> Result<Task>
pub fn set_status(&self, task_id, status) -> Result<()>
pub fn append_output(&self, task_id, output) -> Result<()>
pub fn stop(&self, task_id) -> Result<Task>
pub fn assign_team(&self, task_id, team_id) -> Result<()>
```

---

## Policy Engine

`policy_engine.rs` — Rules for automated decision-making.

```rust
pub struct PolicyRule {
    name: String,
    condition: PolicyCondition,  // And/Or combinators, GreenAt, StaleBranch, etc.
    action: PolicyAction,        // MergeToDev, RecoverOnce, Escalate, Block, Chain
    priority: u32,
}
```

This enables autonomous workflows: "if tests pass at level X, merge to dev"
or "if branch is stale, rebase and retry once before escalating."
