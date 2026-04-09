//! In-process subagent spawning for parallel task delegation.
//!
//! A parent ConversationRuntime can spawn child runtimes with focused tasks,
//! each running with their own session and mode.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::skills::SubagentMode;

/// A task to be executed by a subagent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentTask {
    pub id: String,
    pub mode: SubagentMode,
    pub prompt: String,
    pub parent_session_id: String,
    pub timeout: Duration,
}

/// Result of a subagent's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub output: String,
    pub tool_calls: Vec<String>,
    pub tokens_used: u64,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Completed,
    Failed(String),
    TimedOut,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Completed => write!(f, "completed"),
            Self::Failed(e) => write!(f, "failed: {e}"),
            Self::TimedOut => write!(f, "timed out"),
        }
    }
}

/// Configuration for subagent spawning.
#[derive(Debug, Clone)]
pub struct SubagentConfig {
    /// Maximum concurrent subagents.
    pub max_concurrent: usize,
    /// Maximum iterations for child runtimes (lower than parent).
    pub child_max_iterations: usize,
    /// Default timeout per subagent task.
    pub default_timeout: Duration,
    /// Whether subagents can spawn their own subagents (depth limit).
    pub allow_nested: bool,
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 3,
            child_max_iterations: 10,
            default_timeout: Duration::from_secs(300),
            allow_nested: false,
        }
    }
}

/// Manages subagent spawning and tracks active tasks.
pub struct SubagentSpawner {
    config: SubagentConfig,
    active_count: usize,
    completed: Vec<SubagentResult>,
    is_child: bool,
}

impl SubagentSpawner {
    pub fn new(config: SubagentConfig) -> Self {
        Self {
            config,
            active_count: 0,
            completed: Vec::new(),
            is_child: false,
        }
    }

    /// Create a spawner for a child runtime (can't spawn further).
    pub fn child_spawner() -> Self {
        Self {
            config: SubagentConfig::default(),
            active_count: 0,
            completed: Vec::new(),
            is_child: true,
        }
    }

    /// Check if a new subagent can be spawned.
    pub fn can_spawn(&self) -> bool {
        if self.is_child && !self.config.allow_nested {
            debug!("child spawner cannot spawn nested subagents");
            return false;
        }
        if self.active_count >= self.config.max_concurrent {
            warn!(
                "at max concurrent subagents ({}/{})",
                self.active_count, self.config.max_concurrent
            );
            return false;
        }
        true
    }

    /// Create a SubagentTask with defaults.
    pub fn create_task(
        &self,
        mode: SubagentMode,
        prompt: impl Into<String>,
        parent_session_id: impl Into<String>,
    ) -> SubagentTask {
        SubagentTask {
            id: uuid::Uuid::new_v4().to_string(),
            mode,
            prompt: prompt.into(),
            parent_session_id: parent_session_id.into(),
            timeout: self.config.default_timeout,
        }
    }

    /// Mark a subagent as started (increment active count).
    pub fn on_start(&mut self) {
        self.active_count += 1;
        info!("subagent started ({}/{} active)", self.active_count, self.config.max_concurrent);
    }

    /// Mark a subagent as finished (decrement active count, record result).
    pub fn on_complete(&mut self, result: SubagentResult) {
        self.active_count = self.active_count.saturating_sub(1);
        info!(
            "subagent {} {} ({}/{} active)",
            result.task_id,
            result.status,
            self.active_count,
            self.config.max_concurrent,
        );
        self.completed.push(result);
    }

    /// Get the number of active subagents.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Get completed results.
    pub fn completed(&self) -> &[SubagentResult] {
        &self.completed
    }

    /// Get the child max iterations setting.
    pub fn child_max_iterations(&self) -> usize {
        self.config.child_max_iterations
    }

    /// Get the default timeout.
    pub fn default_timeout(&self) -> Duration {
        self.config.default_timeout
    }
}

/// Format a SubagentResult for injection into the parent's context.
pub fn format_result_for_context(result: &SubagentResult) -> String {
    let tool_summary = if result.tool_calls.is_empty() {
        "no tools used".to_string()
    } else {
        format!("tools: {}", result.tool_calls.join(", "))
    };

    format!(
        "[Subagent {}] Status: {} | {} | tokens: {}\n\n{}",
        result.task_id,
        result.status,
        tool_summary,
        result.tokens_used,
        result.output,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = SubagentConfig::default();
        assert_eq!(config.max_concurrent, 3);
        assert_eq!(config.child_max_iterations, 10);
        assert!(!config.allow_nested);
    }

    #[test]
    fn spawner_tracks_active_count() {
        let mut spawner = SubagentSpawner::new(SubagentConfig {
            max_concurrent: 2,
            ..Default::default()
        });

        assert!(spawner.can_spawn());
        spawner.on_start();
        assert!(spawner.can_spawn());
        spawner.on_start();
        assert!(!spawner.can_spawn()); // at max

        spawner.on_complete(SubagentResult {
            task_id: "t1".to_string(),
            status: TaskStatus::Completed,
            output: "done".to_string(),
            tool_calls: vec![],
            tokens_used: 100,
            completed_at: Utc::now(),
        });
        assert!(spawner.can_spawn()); // back to 1
        assert_eq!(spawner.completed().len(), 1);
    }

    #[test]
    fn child_cannot_spawn() {
        let spawner = SubagentSpawner::child_spawner();
        assert!(!spawner.can_spawn());
    }

    #[test]
    fn create_task_with_defaults() {
        let spawner = SubagentSpawner::new(SubagentConfig::default());
        let task = spawner.create_task(
            SubagentMode::CodeGenerator,
            "Write augmentation code",
            "parent-123",
        );
        assert_eq!(task.mode, SubagentMode::CodeGenerator);
        assert_eq!(task.parent_session_id, "parent-123");
        assert_eq!(task.timeout, Duration::from_secs(300));
    }

    #[test]
    fn format_result() {
        let result = SubagentResult {
            task_id: "t1".to_string(),
            status: TaskStatus::Completed,
            output: "Generated the pipeline code.".to_string(),
            tool_calls: vec!["write_file".to_string(), "run_command".to_string()],
            tokens_used: 5000,
            completed_at: Utc::now(),
        };
        let formatted = format_result_for_context(&result);
        assert!(formatted.contains("write_file, run_command"));
        assert!(formatted.contains("Generated the pipeline code"));
    }
}
