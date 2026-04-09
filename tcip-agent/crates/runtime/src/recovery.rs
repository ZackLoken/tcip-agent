use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Classifies known failure scenarios for automated recovery.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureScenario {
    /// MCP server subprocess crashed or disconnected.
    McpServerCrash,
    /// CUDA out-of-memory during training/inference.
    GpuOom,
    /// YOLO annotation file has unparseable lines.
    CorruptAnnotation,
    /// Prediction file is older than the model checkpoint.
    StalePredictions,
    /// MCP registry tools are unreachable.
    RegistryUnreachable,
    /// Training loss diverged (NaN or extremely large).
    TrainingDiverged,
    /// Generic tool execution failure (no specific recipe).
    Unknown,
}

/// What to do when max recovery attempts are exhausted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscalationPolicy {
    /// Abort the current turn entirely.
    Abort,
    /// Alert human via checkpoint and wait for guidance.
    AlertHuman,
    /// Log the error and continue without the tool result.
    LogAndContinue,
}

/// A single step in the recovery process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStep {
    pub description: String,
    pub action: RecoveryAction,
}

/// Concrete recovery action to take.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    /// Inject a system message hinting the model about the failure and how to retry.
    InjectHint(String),
    /// Suggest the model modify a parameter before retrying.
    SuggestParameterChange { param: String, suggestion: String },
    /// Clear cached data (e.g. stale predictions).
    ClearCache { description: String },
    /// Log the error and skip this tool call.
    LogAndSkip,
}

/// A recovery recipe matching a failure scenario to corrective steps.
#[derive(Debug, Clone)]
pub struct RecoveryRecipe {
    pub scenario: FailureScenario,
    pub steps: Vec<RecoveryStep>,
    pub max_attempts: u32,
    pub escalation: EscalationPolicy,
}

/// Result of attempting recovery.
#[derive(Debug)]
pub enum RecoveryResult {
    /// Recovery succeeded — the model should retry with the hint/suggestion.
    Retry { hint: String },
    /// Recovery exhausted — escalate per policy.
    Escalate { scenario: FailureScenario, policy: EscalationPolicy },
    /// No matching recipe — return original error.
    NoRecipe,
}

/// Engine that matches errors to recovery recipes and tracks attempts.
pub struct RecoveryEngine {
    recipes: Vec<RecoveryRecipe>,
    /// Track attempts per scenario in the current turn.
    attempt_counts: std::collections::HashMap<FailureScenario, u32>,
}

impl RecoveryEngine {
    /// Create a recovery engine with the default ML/CV recipes.
    pub fn new() -> Self {
        Self {
            recipes: default_recipes(),
            attempt_counts: std::collections::HashMap::new(),
        }
    }

    /// Try to recover from a tool error.
    ///
    /// Returns a `RecoveryResult` indicating whether to retry, escalate, or pass through.
    pub fn try_recover(&mut self, tool_name: &str, error: &str) -> RecoveryResult {
        let scenario = classify_error(tool_name, error);

        if scenario == FailureScenario::Unknown {
            return RecoveryResult::NoRecipe;
        }

        let recipe = match self.recipes.iter().find(|r| r.scenario == scenario) {
            Some(r) => r.clone(),
            None => return RecoveryResult::NoRecipe,
        };

        let count = self.attempt_counts.entry(scenario.clone()).or_insert(0);
        *count += 1;

        if *count > recipe.max_attempts {
            warn!(
                "recovery exhausted for {:?} after {} attempts, escalating",
                scenario, count
            );
            return RecoveryResult::Escalate {
                scenario,
                policy: recipe.escalation,
            };
        }

        info!(
            "recovery attempt {}/{} for {:?}",
            count, recipe.max_attempts, scenario
        );

        // Build a combined hint from all recipe steps
        let hint = recipe
            .steps
            .iter()
            .map(|step| match &step.action {
                RecoveryAction::InjectHint(h) => h.clone(),
                RecoveryAction::SuggestParameterChange { param, suggestion } => {
                    format!("Try changing {param}: {suggestion}")
                }
                RecoveryAction::ClearCache { description } => {
                    format!("Clear cache: {description}")
                }
                RecoveryAction::LogAndSkip => "Skip this operation and continue.".to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ");

        RecoveryResult::Retry { hint }
    }

    /// Reset attempt counters (call at turn boundaries).
    pub fn reset(&mut self) {
        self.attempt_counts.clear();
    }
}

impl Default for RecoveryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Classify an error string into a known failure scenario.
fn classify_error(tool_name: &str, error: &str) -> FailureScenario {
    let error_lower = error.to_lowercase();

    // MCP connection errors
    if error_lower.contains("connection refused")
        || error_lower.contains("broken pipe")
        || error_lower.contains("mcp")
            && (error_lower.contains("crash") || error_lower.contains("disconnect"))
    {
        return FailureScenario::McpServerCrash;
    }

    // GPU OOM
    if error_lower.contains("cuda out of memory")
        || error_lower.contains("out of memory")
            && error_lower.contains("cuda")
        || error_lower.contains("oom")
            && error_lower.contains("gpu")
    {
        return FailureScenario::GpuOom;
    }

    // Corrupt annotations
    if (error_lower.contains("label") || error_lower.contains("annotation"))
        && (error_lower.contains("parse") || error_lower.contains("malformed") || error_lower.contains("invalid format"))
    {
        return FailureScenario::CorruptAnnotation;
    }

    // Stale predictions
    if error_lower.contains("prediction") && error_lower.contains("stale")
        || error_lower.contains("prediction") && error_lower.contains("older than")
    {
        return FailureScenario::StalePredictions;
    }

    // Registry unreachable
    if tool_name.starts_with("mcp__")
        && (tool_name.contains("registry") || tool_name.contains("crop") || tool_name.contains("trait"))
        && (error_lower.contains("timeout") || error_lower.contains("unreachable") || error_lower.contains("connection"))
    {
        return FailureScenario::RegistryUnreachable;
    }

    // Training diverged
    if error_lower.contains("nan") && error_lower.contains("loss")
        || error_lower.contains("diverge")
        || error_lower.contains("loss") && error_lower.contains("inf")
    {
        return FailureScenario::TrainingDiverged;
    }

    FailureScenario::Unknown
}

/// Build the default set of ML/CV recovery recipes.
fn default_recipes() -> Vec<RecoveryRecipe> {
    vec![
        RecoveryRecipe {
            scenario: FailureScenario::McpServerCrash,
            steps: vec![RecoveryStep {
                description: "Restart MCP subprocess and retry".to_string(),
                action: RecoveryAction::InjectHint(
                    "The MCP server crashed. It has been restarted. \
                     Please retry the last tool call."
                        .to_string(),
                ),
            }],
            max_attempts: 1,
            escalation: EscalationPolicy::Abort,
        },
        RecoveryRecipe {
            scenario: FailureScenario::GpuOom,
            steps: vec![RecoveryStep {
                description: "Reduce batch size and retry".to_string(),
                action: RecoveryAction::SuggestParameterChange {
                    param: "batch_size".to_string(),
                    suggestion: "Halve the current batch size and retry training.".to_string(),
                },
            }],
            max_attempts: 2,
            escalation: EscalationPolicy::AlertHuman,
        },
        RecoveryRecipe {
            scenario: FailureScenario::CorruptAnnotation,
            steps: vec![RecoveryStep {
                description: "Skip malformed lines and report count".to_string(),
                action: RecoveryAction::InjectHint(
                    "Some annotation lines are malformed. Skip invalid lines, \
                     report how many were skipped, and continue with valid data."
                        .to_string(),
                ),
            }],
            max_attempts: 1,
            escalation: EscalationPolicy::LogAndContinue,
        },
        RecoveryRecipe {
            scenario: FailureScenario::StalePredictions,
            steps: vec![RecoveryStep {
                description: "Clear prediction cache and re-run inference".to_string(),
                action: RecoveryAction::ClearCache {
                    description: "Delete stale prediction files and re-run inference with current model."
                        .to_string(),
                },
            }],
            max_attempts: 1,
            escalation: EscalationPolicy::LogAndContinue,
        },
        RecoveryRecipe {
            scenario: FailureScenario::RegistryUnreachable,
            steps: vec![RecoveryStep {
                description: "Fall back to cached crops.yml".to_string(),
                action: RecoveryAction::InjectHint(
                    "The MCP registry is unreachable. Fall back to the local \
                     registry/crops.yml file for trait information."
                        .to_string(),
                ),
            }],
            max_attempts: 1,
            escalation: EscalationPolicy::LogAndContinue,
        },
        RecoveryRecipe {
            scenario: FailureScenario::TrainingDiverged,
            steps: vec![RecoveryStep {
                description: "Reduce learning rate and retry from last checkpoint".to_string(),
                action: RecoveryAction::SuggestParameterChange {
                    param: "learning_rate".to_string(),
                    suggestion: "Reduce LR by 10x and resume from the last saved checkpoint."
                        .to_string(),
                },
            }],
            max_attempts: 2,
            escalation: EscalationPolicy::AlertHuman,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_gpu_oom() {
        let scenario = classify_error("mcp__launch_training", "RuntimeError: CUDA out of memory");
        assert_eq!(scenario, FailureScenario::GpuOom);
    }

    #[test]
    fn classifies_mcp_crash() {
        let scenario = classify_error("mcp__list_crops", "Connection refused");
        assert_eq!(scenario, FailureScenario::McpServerCrash);
    }

    #[test]
    fn classifies_corrupt_annotation() {
        let scenario = classify_error("mcp__load_labels", "Label parse error: malformed line 42");
        assert_eq!(scenario, FailureScenario::CorruptAnnotation);
    }

    #[test]
    fn classifies_training_diverged() {
        let scenario = classify_error("mcp__launch_training", "Training stopped: loss is NaN at epoch 5");
        assert_eq!(scenario, FailureScenario::TrainingDiverged);
    }

    #[test]
    fn unknown_error_returns_no_recipe() {
        let mut engine = RecoveryEngine::new();
        match engine.try_recover("some_tool", "unexpected segfault") {
            RecoveryResult::NoRecipe => {}
            other => panic!("expected NoRecipe, got {other:?}"),
        }
    }

    #[test]
    fn recovery_succeeds_then_escalates() {
        let mut engine = RecoveryEngine::new();
        // First attempt should retry
        match engine.try_recover("mcp__launch_training", "CUDA out of memory") {
            RecoveryResult::Retry { .. } => {}
            other => panic!("expected Retry, got {other:?}"),
        }
        // Second attempt should also retry (max_attempts=2)
        match engine.try_recover("mcp__launch_training", "CUDA out of memory") {
            RecoveryResult::Retry { .. } => {}
            other => panic!("expected Retry, got {other:?}"),
        }
        // Third attempt should escalate
        match engine.try_recover("mcp__launch_training", "CUDA out of memory") {
            RecoveryResult::Escalate { policy, .. } => {
                assert_eq!(policy, EscalationPolicy::AlertHuman);
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    #[test]
    fn reset_clears_attempt_counts() {
        let mut engine = RecoveryEngine::new();
        // Use up the single attempt for MCP crash
        engine.try_recover("mcp__list_crops", "Connection refused");
        // Should now escalate
        match engine.try_recover("mcp__list_crops", "Connection refused") {
            RecoveryResult::Escalate { .. } => {}
            other => panic!("expected Escalate, got {other:?}"),
        }
        // Reset and try again — should succeed
        engine.reset();
        match engine.try_recover("mcp__list_crops", "Connection refused") {
            RecoveryResult::Retry { .. } => {}
            other => panic!("expected Retry after reset, got {other:?}"),
        }
    }
}
