use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Three-tier permission model for TCIP agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PermissionMode {
    /// Registry queries, file reads, grep — safe browsing.
    ReadOnly,
    /// File writes, annotations, inference — normal work.
    WorkspaceWrite,
    /// Training launch, HPO, model deployment — needs explicit approval.
    FullAccess,
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnly => write!(f, "ReadOnly"),
            Self::WorkspaceWrite => write!(f, "WorkspaceWrite"),
            Self::FullAccess => write!(f, "FullAccess"),
        }
    }
}

/// Result of a permission check.
#[derive(Debug, Clone)]
pub enum PermissionResult {
    /// Tool execution is allowed.
    Allowed,
    /// Requires user approval via HITL checkpoint.
    NeedsApproval { tool: String, description: String },
    /// Hard denied.
    Denied { reason: String },
}

/// Enforces permission policy on tool execution.
pub struct PermissionEnforcer {
    active_mode: PermissionMode,
    tool_requirements: BTreeMap<String, PermissionMode>,
}

impl PermissionEnforcer {
    #[must_use]
    pub fn new(mode: PermissionMode) -> Self {
        let mut tool_requirements = BTreeMap::new();

        // Native tools
        tool_requirements.insert("read_file".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("grep_search".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("glob_search".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("write_file".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("edit_file".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("bash".to_string(), PermissionMode::FullAccess);

        // MCP tools — domain-specific permission mapping
        tool_requirements.insert("mcp__list_crops".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__get_crop_traits".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__get_trait_info".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__find_traits_by_task".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__find_traits_by_sensor".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__get_registry_summary".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__load_dataset".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__validate_data_quality".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__load_annotations".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__evaluate_detections".to_string(), PermissionMode::ReadOnly);
        tool_requirements.insert("mcp__evaluate_dataset".to_string(), PermissionMode::ReadOnly);

        tool_requirements.insert("mcp__split_dataset".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("mcp__save_annotations".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("mcp__run_inference".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("mcp__export_predictions_yolo".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("mcp__export_results_csv".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("mcp__init_project".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("mcp__create_session".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("mcp__append_session_event".to_string(), PermissionMode::WorkspaceWrite);
        tool_requirements.insert("mcp__register_model".to_string(), PermissionMode::WorkspaceWrite);

        tool_requirements.insert("mcp__launch_training".to_string(), PermissionMode::FullAccess);
        tool_requirements.insert("mcp__run_hpo".to_string(), PermissionMode::FullAccess);

        Self {
            active_mode: mode,
            tool_requirements,
        }
    }

    /// Check if a tool call is permitted under the current mode.
    #[must_use]
    pub fn check(&self, tool_name: &str) -> PermissionResult {
        let required = self
            .tool_requirements
            .get(tool_name)
            .copied()
            // Unknown MCP tools default to WorkspaceWrite
            .unwrap_or_else(|| {
                if tool_name.starts_with("mcp__") {
                    PermissionMode::WorkspaceWrite
                } else {
                    PermissionMode::FullAccess
                }
            });

        if self.active_mode >= required {
            PermissionResult::Allowed
        } else if required == PermissionMode::FullAccess {
            PermissionResult::NeedsApproval {
                tool: tool_name.to_string(),
                description: format!(
                    "{tool_name} requires FullAccess; current mode is {}",
                    self.active_mode
                ),
            }
        } else {
            PermissionResult::Denied {
                reason: format!(
                    "{tool_name} requires {required}; current mode is {}",
                    self.active_mode
                ),
            }
        }
    }

    /// Get the active permission mode.
    #[must_use]
    pub fn mode(&self) -> PermissionMode {
        self.active_mode
    }

    /// Update the active permission mode.
    pub fn set_mode(&mut self, mode: PermissionMode) {
        self.active_mode = mode;
    }
}
