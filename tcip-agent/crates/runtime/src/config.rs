use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 3-level configuration: defaults → project → session/CLI overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub permission_mode: String,
    pub mcp_command: Option<String>,
    pub mcp_args: Vec<String>,
    pub workspace_root: PathBuf,
    pub sessions_dir: PathBuf,
    pub skills_dir: PathBuf,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            temperature: 0.0,
            permission_mode: "WorkspaceWrite".to_string(),
            mcp_command: None,
            mcp_args: Vec::new(),
            workspace_root: PathBuf::from("."),
            sessions_dir: PathBuf::from(".tcip/sessions"),
            skills_dir: PathBuf::from("skills"),
        }
    }
}

impl RuntimeConfig {
    /// Load config from defaults → project `.tcip/config.toml` → environment overrides.
    pub fn load(workspace: &std::path::Path) -> Self {
        let mut config = Self::default();
        config.workspace_root = workspace.to_path_buf();
        config.sessions_dir = workspace.join(".tcip").join("sessions");
        config.skills_dir = workspace.join("skills");

        // Try loading project config
        let project_config_path = workspace.join(".tcip").join("config.toml");
        if let Ok(content) = std::fs::read_to_string(&project_config_path) {
            if let Ok(table) = content.parse::<toml::Table>() {
                if let Some(model) = table.get("model").and_then(|v| v.as_str()) {
                    config.model = model.to_string();
                }
                if let Some(max_tokens) = table.get("max_tokens").and_then(|v| v.as_integer()) {
                    config.max_tokens = max_tokens as u32;
                }
                if let Some(mode) = table.get("permission_mode").and_then(|v| v.as_str()) {
                    config.permission_mode = mode.to_string();
                }
                if let Some(cmd) = table.get("mcp_command").and_then(|v| v.as_str()) {
                    config.mcp_command = Some(cmd.to_string());
                }
                if let Some(args) = table.get("mcp_args").and_then(|v| v.as_array()) {
                    config.mcp_args = args
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                }
            }
        }

        // Environment variable overrides
        if let Ok(model) = std::env::var("TCIP_MODEL") {
            config.model = model;
        }
        if let Ok(mode) = std::env::var("TCIP_PERMISSION_MODE") {
            config.permission_mode = mode;
        }

        config
    }

    /// Resolve the permission mode from string.
    #[must_use]
    pub fn resolved_permission_mode(&self) -> crate::permission::PermissionMode {
        match self.permission_mode.as_str() {
            "ReadOnly" => crate::permission::PermissionMode::ReadOnly,
            "FullAccess" => crate::permission::PermissionMode::FullAccess,
            _ => crate::permission::PermissionMode::WorkspaceWrite,
        }
    }
}
