use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::debug;

/// Result of a pre-tool-use hook.
#[derive(Debug, Default)]
pub struct HookResult {
    /// If false, the tool call is blocked.
    pub proceed: bool,
    /// Optionally replace the tool input.
    pub updated_input: Option<String>,
    /// Messages to inject into context (e.g. warnings, hints).
    pub messages: Vec<String>,
}

impl HookResult {
    /// A result that allows the tool call to proceed unchanged.
    pub fn allow() -> Self {
        Self {
            proceed: true,
            updated_input: None,
            messages: Vec::new(),
        }
    }

    /// A result that blocks the tool call with a message.
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            proceed: false,
            updated_input: None,
            messages: vec![reason.into()],
        }
    }
}

/// Result of a post-tool-use hook.
#[derive(Debug, Default)]
pub struct PostHookResult {
    /// Feedback messages appended to the tool result context.
    pub feedback: Vec<String>,
}

/// Result of a post-tool-failure hook.
#[derive(Debug, Default)]
pub struct FailureHookResult {
    /// Optional recovery hint for the recovery engine.
    pub recovery_hint: Option<String>,
}

/// Trait for tool lifecycle hooks.
///
/// Implementations can inspect and modify tool calls before execution,
/// capture results after execution, and provide hints on failure.
pub trait ToolHook: Send + Sync {
    /// Called before a tool is executed. Can block or modify the call.
    fn pre_tool_use(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> HookResult;

    /// Called after a tool executes successfully.
    fn post_tool_use(
        &self,
        _tool_name: &str,
        _input: &serde_json::Value,
        _output: &str,
    ) -> PostHookResult {
        PostHookResult::default()
    }

    /// Called after a tool execution fails.
    fn post_tool_use_failure(
        &self,
        _tool_name: &str,
        _input: &serde_json::Value,
        _error: &str,
    ) -> FailureHookResult {
        FailureHookResult::default()
    }

    /// A human-readable name for this hook (for logging).
    fn name(&self) -> &str;
}

/// Abort signal shared between hooks and the runtime.
///
/// Any hook can set this to trigger an abort of the current operation.
#[derive(Debug, Clone)]
pub struct AbortSignal {
    inner: Arc<AtomicBool>,
}

impl AbortSignal {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn abort(&self) {
        self.inner.store(true, Ordering::Release);
    }

    pub fn is_aborted(&self) -> bool {
        self.inner.load(Ordering::Acquire)
    }

    pub fn reset(&self) {
        self.inner.store(false, Ordering::Release);
    }
}

impl Default for AbortSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs a chain of hooks in registration order.
///
/// Pre-hooks are run sequentially; if any returns `proceed: false`,
/// subsequent hooks are skipped and the tool call is blocked.
pub struct HookRunner {
    hooks: Vec<Box<dyn ToolHook>>,
    abort_signal: AbortSignal,
}

impl HookRunner {
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            abort_signal: AbortSignal::new(),
        }
    }

    /// Register a hook.
    pub fn register(&mut self, hook: Box<dyn ToolHook>) {
        debug!("registered hook: {}", hook.name());
        self.hooks.push(hook);
    }

    /// Get a clone of the abort signal for external use.
    pub fn abort_signal(&self) -> AbortSignal {
        self.abort_signal.clone()
    }

    /// Run all pre-tool hooks. Returns combined result.
    ///
    /// If any hook blocks the call, returns immediately with `proceed: false`.
    pub fn run_pre_hooks(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> HookResult {
        if self.abort_signal.is_aborted() {
            return HookResult::block("Operation aborted by signal");
        }

        let mut combined_messages = Vec::new();
        let mut final_input = None;

        for hook in &self.hooks {
            let result = hook.pre_tool_use(tool_name, input);
            combined_messages.extend(result.messages);

            if !result.proceed {
                debug!("hook '{}' blocked tool '{}'", hook.name(), tool_name);
                return HookResult {
                    proceed: false,
                    updated_input: None,
                    messages: combined_messages,
                };
            }

            if result.updated_input.is_some() {
                final_input = result.updated_input;
            }
        }

        HookResult {
            proceed: true,
            updated_input: final_input,
            messages: combined_messages,
        }
    }

    /// Run all post-tool hooks. Returns combined feedback.
    pub fn run_post_hooks(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        output: &str,
    ) -> PostHookResult {
        let mut feedback = Vec::new();
        for hook in &self.hooks {
            let result = hook.post_tool_use(tool_name, input, output);
            feedback.extend(result.feedback);
        }
        PostHookResult { feedback }
    }

    /// Run all failure hooks. Returns first recovery hint found.
    pub fn run_failure_hooks(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        error: &str,
    ) -> FailureHookResult {
        for hook in &self.hooks {
            let result = hook.post_tool_use_failure(tool_name, input, error);
            if result.recovery_hint.is_some() {
                return result;
            }
        }
        FailureHookResult::default()
    }

    /// Check if any hooks are registered.
    pub fn has_hooks(&self) -> bool {
        !self.hooks.is_empty()
    }
}

impl Default for HookRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct AllowHook;
    impl ToolHook for AllowHook {
        fn pre_tool_use(&self, _: &str, _: &serde_json::Value) -> HookResult {
            HookResult::allow()
        }
        fn name(&self) -> &str {
            "allow_hook"
        }
    }

    struct BlockHook {
        reason: String,
    }
    impl ToolHook for BlockHook {
        fn pre_tool_use(&self, _: &str, _: &serde_json::Value) -> HookResult {
            HookResult::block(&self.reason)
        }
        fn name(&self) -> &str {
            "block_hook"
        }
    }

    struct FeedbackHook;
    impl ToolHook for FeedbackHook {
        fn pre_tool_use(&self, _: &str, _: &serde_json::Value) -> HookResult {
            HookResult::allow()
        }
        fn post_tool_use(&self, _: &str, _: &serde_json::Value, _: &str) -> PostHookResult {
            PostHookResult {
                feedback: vec!["artifact logged".to_string()],
            }
        }
        fn name(&self) -> &str {
            "feedback_hook"
        }
    }

    #[test]
    fn allow_hook_proceeds() {
        let mut runner = HookRunner::new();
        runner.register(Box::new(AllowHook));
        let result = runner.run_pre_hooks("read_file", &json!({}));
        assert!(result.proceed);
    }

    #[test]
    fn block_hook_stops_execution() {
        let mut runner = HookRunner::new();
        runner.register(Box::new(BlockHook {
            reason: "no GPU".to_string(),
        }));
        let result = runner.run_pre_hooks("mcp__launch_training", &json!({}));
        assert!(!result.proceed);
        assert!(result.messages.iter().any(|m| m.contains("no GPU")));
    }

    #[test]
    fn block_hook_short_circuits_chain() {
        let mut runner = HookRunner::new();
        runner.register(Box::new(BlockHook {
            reason: "blocked".to_string(),
        }));
        runner.register(Box::new(AllowHook)); // Should never run
        let result = runner.run_pre_hooks("tool", &json!({}));
        assert!(!result.proceed);
    }

    #[test]
    fn post_hooks_collect_feedback() {
        let mut runner = HookRunner::new();
        runner.register(Box::new(FeedbackHook));
        let result = runner.run_post_hooks("mcp__run_inference", &json!({}), "ok");
        assert_eq!(result.feedback.len(), 1);
        assert!(result.feedback[0].contains("artifact logged"));
    }

    #[test]
    fn abort_signal_blocks_all() {
        let mut runner = HookRunner::new();
        runner.register(Box::new(AllowHook));
        runner.abort_signal().abort();
        let result = runner.run_pre_hooks("any_tool", &json!({}));
        assert!(!result.proceed);
    }

    #[test]
    fn abort_signal_reset() {
        let runner = HookRunner::new();
        let signal = runner.abort_signal();
        signal.abort();
        assert!(signal.is_aborted());
        signal.reset();
        assert!(!signal.is_aborted());
    }

    #[test]
    fn no_hooks_allows_all() {
        let runner = HookRunner::new();
        let result = runner.run_pre_hooks("any_tool", &json!({}));
        assert!(result.proceed);
        assert!(!runner.has_hooks());
    }
}
