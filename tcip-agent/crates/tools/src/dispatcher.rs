use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::bash;
use crate::file_ops;
use crate::mcp_bridge::McpBridge;
use crate::search;

// ── Tool specification ──

/// Declares a tool's schema for the Anthropic API.
#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub required_permission: PermissionLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PermissionLevel {
    ReadOnly,
    WorkspaceWrite,
    FullAccess,
}

// ── Errors ──

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("permission denied: {tool} requires {required:?}, current mode is {current:?}")]
    PermissionDenied {
        tool: String,
        required: PermissionLevel,
        current: PermissionLevel,
    },
    #[error("MCP error: {0}")]
    Mcp(String),
}

// ── Executor trait ──

/// Trait for tool execution, enabling mock implementations in tests.
pub trait ToolExecutor: Send {
    fn execute(
        &mut self,
        tool_name: &str,
        input: &Value,
    ) -> impl std::future::Future<Output = Result<String, ToolError>> + Send;

    fn list_tools(&self) -> Vec<ToolSpec>;
}

// ── Dispatcher ──

/// Routes tool calls to native implementations or MCP bridge.
pub struct ToolDispatcher {
    workspace_root: std::path::PathBuf,
    mcp_bridge: Option<McpBridge>,
    mcp_tools: BTreeMap<String, ToolSpec>,
}

impl ToolDispatcher {
    #[must_use]
    pub fn new(workspace_root: std::path::PathBuf) -> Self {
        Self {
            workspace_root,
            mcp_bridge: None,
            mcp_tools: BTreeMap::new(),
        }
    }

    /// Connect to an MCP server and discover its tools.
    pub async fn connect_mcp(&mut self, bridge: McpBridge) -> Result<(), ToolError> {
        let tools = bridge.list_tools().await?;
        for spec in tools {
            self.mcp_tools
                .insert(format!("mcp__{}", spec.name), spec);
        }
        self.mcp_bridge = Some(bridge);
        Ok(())
    }

    /// Get the number of registered MCP tools.
    #[must_use]
    pub fn mcp_tool_count(&self) -> usize {
        self.mcp_tools.len()
    }

    fn native_tools() -> Vec<ToolSpec> {
        let mut tools = Vec::new();
        tools.extend(file_ops::tool_specs());
        tools.extend(bash::tool_specs());
        tools.extend(search::tool_specs());
        tools
    }

    fn is_native(name: &str) -> bool {
        matches!(
            name,
            "read_file"
                | "write_file"
                | "edit_file"
                | "bash"
                | "grep_search"
                | "glob_search"
        )
    }

    async fn execute_native(
        &self,
        name: &str,
        input: &Value,
    ) -> Result<String, ToolError> {
        match name {
            "read_file" => file_ops::read_file(input, &self.workspace_root),
            "write_file" => file_ops::write_file(input, &self.workspace_root),
            "edit_file" => file_ops::edit_file(input, &self.workspace_root),
            "bash" => bash::run_bash(input, &self.workspace_root).await,
            "grep_search" => search::grep_search(input, &self.workspace_root),
            "glob_search" => search::glob_search(input, &self.workspace_root),
            _ => Err(ToolError::UnknownTool(name.to_string())),
        }
    }

    async fn execute_mcp(
        &self,
        name: &str,
        input: &Value,
    ) -> Result<String, ToolError> {
        let bridge = self
            .mcp_bridge
            .as_ref()
            .ok_or_else(|| ToolError::Mcp("MCP not connected".to_string()))?;

        // Strip mcp__ prefix to get original tool name
        let mcp_name = name.strip_prefix("mcp__").unwrap_or(name);
        bridge.call_tool(mcp_name, input).await
    }
}

impl ToolExecutor for ToolDispatcher {
    async fn execute(
        &mut self,
        tool_name: &str,
        input: &Value,
    ) -> Result<String, ToolError> {
        if Self::is_native(tool_name) {
            self.execute_native(tool_name, input).await
        } else if self.mcp_tools.contains_key(tool_name) {
            self.execute_mcp(tool_name, input).await
        } else {
            Err(ToolError::UnknownTool(tool_name.to_string()))
        }
    }

    fn list_tools(&self) -> Vec<ToolSpec> {
        let mut tools = Self::native_tools();
        tools.extend(self.mcp_tools.values().cloned());
        tools
    }
}
