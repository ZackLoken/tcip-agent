use serde::{Deserialize, Serialize};

/// HITL checkpoint types for the 5 workflow gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Checkpoint {
    /// Show proposed pipeline config, await approval.
    PipelineDesign {
        description: String,
        config: serde_json::Value,
    },
    /// Show training config + augmentation, await review.
    ConfigReview {
        description: String,
        config: serde_json::Value,
    },
    /// Show config + cost estimate, await confirmation.
    TrainingLaunch {
        tool_name: String,
        tool_input: serde_json::Value,
        estimated_cost: Option<String>,
    },
    /// Show metrics + samples, await accept/retrain.
    ResultsReview {
        metrics: serde_json::Value,
        sample_images: Vec<String>,
    },
    /// Show final model + test results, await confirmation.
    ModelDeployment {
        model_info: serde_json::Value,
        test_results: serde_json::Value,
    },
}

/// User's response to a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointResolution {
    Approved,
    Denied { reason: String },
    Modified { changes: serde_json::Value },
}

/// Trait for resolving checkpoints — swap between stdin REPL and GUI dialog.
pub trait CheckpointResolver: Send {
    fn resolve(
        &self,
        checkpoint: &Checkpoint,
    ) -> impl std::future::Future<Output = CheckpointResolution> + Send;
}

/// Simple stdin-based resolver for Phase 2 REPL.
pub struct StdinCheckpointResolver;

impl CheckpointResolver for StdinCheckpointResolver {
    async fn resolve(&self, checkpoint: &Checkpoint) -> CheckpointResolution {
        let description = match checkpoint {
            Checkpoint::PipelineDesign { description, config } => {
                eprintln!("\n═══ CHECKPOINT: Pipeline Design ═══");
                eprintln!("{description}");
                eprintln!(
                    "Config: {}",
                    serde_json::to_string_pretty(config).unwrap_or_default()
                );
                "pipeline design"
            }
            Checkpoint::ConfigReview { description, config } => {
                eprintln!("\n═══ CHECKPOINT: Config Review ═══");
                eprintln!("{description}");
                eprintln!(
                    "Config: {}",
                    serde_json::to_string_pretty(config).unwrap_or_default()
                );
                "config review"
            }
            Checkpoint::TrainingLaunch {
                tool_name,
                tool_input,
                estimated_cost,
            } => {
                eprintln!("\n═══ CHECKPOINT: Training Launch ═══");
                eprintln!("Tool: {tool_name}");
                eprintln!(
                    "Input: {}",
                    serde_json::to_string_pretty(tool_input).unwrap_or_default()
                );
                if let Some(cost) = estimated_cost {
                    eprintln!("Estimated cost: {cost}");
                }
                "training launch"
            }
            Checkpoint::ResultsReview {
                metrics,
                sample_images,
            } => {
                eprintln!("\n═══ CHECKPOINT: Results Review ═══");
                eprintln!(
                    "Metrics: {}",
                    serde_json::to_string_pretty(metrics).unwrap_or_default()
                );
                eprintln!("Sample images: {}", sample_images.join(", "));
                "results review"
            }
            Checkpoint::ModelDeployment {
                model_info,
                test_results,
            } => {
                eprintln!("\n═══ CHECKPOINT: Model Deployment ═══");
                eprintln!(
                    "Model: {}",
                    serde_json::to_string_pretty(model_info).unwrap_or_default()
                );
                eprintln!(
                    "Test results: {}",
                    serde_json::to_string_pretty(test_results).unwrap_or_default()
                );
                "model deployment"
            }
        };

        eprintln!("Approve {description}? [y/n]: ");

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_ok() {
            let trimmed = input.trim().to_lowercase();
            if trimmed == "y" || trimmed == "yes" {
                return CheckpointResolution::Approved;
            }
        }

        CheckpointResolution::Denied {
            reason: "User denied checkpoint".to_string(),
        }
    }
}
